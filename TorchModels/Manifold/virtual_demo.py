import os
import math
import cv2
import numpy as np
import torch
from bacteria_core import (Config, BacteriaNCA, seed_state, Fields, Teacher,
                           composite, load_checkpoint, COLOURS, PERSON_RGB,
                           RED, ORANGE, BLACK, BLUE, GREEN)

TOOLS = ["red", "orange", "black", "blue", "green", "erase"]
TOOL_LABELS = {"red": "FOOD", "orange": "DECOY", "black": "TOXIN",
               "blue": "WALL", "green": "BOOST", "erase": "ERASE"}
TOOL_H, BRUSH_H = 54, 36

def disc_mask(g, y, x, r):
    yy, xx = np.mgrid[0:g, 0:g]
    return (yy - y) ** 2 + (xx - x) ** 2 <= r * r

class VirtualSession:
    def __init__(self, model, cfg, grid=None, steps_per_frame=2, n=0.5, panel_px=512):
        self.model, self.cfg = model, cfg
        self.g = grid or cfg.grid
        self.dev = next(model.parameters()).device
        self.fields = Fields(cfg, 1, self.g, self.g, self.dev)
        self.canvas = np.zeros((5, self.g, self.g), np.float32)
        self.masks = torch.zeros(1, 5, self.g, self.g, device=self.dev)
        self.steps_per_frame, self.panel_px = steps_per_frame, panel_px
        self.n, self.t, self.state = n, 0, None
        self.seed_cell = (self.g // 2, self.g // 2)
        self._w, self._wn = None, None
        self.environment_view, self.environment_dirty = None, True
        rng = np.random.default_rng(7)
        self.breathe_phase = rng.uniform(0, 2 * math.pi, (self.g, self.g))
        self.breathe_speed = rng.uniform(0.07, 0.15, (self.g, self.g))

    def _weights(self):
        if self._wn != self.n:
            self._w, self._wn = self.model.weights_for(torch.tensor([self.n], device=self.dev)), self.n
        return self._w

    def paint(self, name, gy, gx, r, erase=False):
        d = disc_mask(self.g, gy, gx, r)
        self.canvas[:, d] = 0
        if not erase:
            self.canvas[COLOURS.index(name), d] = 1
        self.environment_dirty = True

    def draw_environment_pixels(self, image):
        """Turn the same logical masks into clean pixel-art environment objects."""
        scale = self.panel_px / self.g
        colours = {RED: (55, 75, 220), ORANGE: (30, 145, 245), BLACK: (48, 43, 50),
                   BLUE: (190, 115, 75), GREEN: (65, 175, 80)}
        for c in (GREEN, ORANGE, RED, BLACK, BLUE):
            ys, xs = np.nonzero(self.canvas[c])
            for gy, gx in zip(ys, xs):
                x0, y0 = int(gx * scale), int(gy * scale)
                x1, y1 = int((gx + 1) * scale), int((gy + 1) * scale)
                cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                if c == BLUE:
                    cv2.rectangle(image, (x0, y0), (x1, y1), colours[c], -1)
                    cv2.rectangle(image, (x0, y0), (x1, y1), (150, 82, 58), 1)
                else:
                    radius = max(2, int(0.38 * scale))
                    cv2.circle(image, (cx, cy), radius, colours[c], -1, cv2.LINE_AA)
                    cv2.circle(image, (cx - radius // 3, cy - radius // 3), max(1, radius // 3),
                               (235, 245, 235), -1, cv2.LINE_AA)

    def environment_image(self):
        if self.environment_dirty or self.environment_view is None:
            self.environment_view = np.full((self.panel_px, self.panel_px, 3), 255, np.uint8)
            self.draw_environment_pixels(self.environment_view)
            self.environment_dirty = False
        return self.environment_view

    def reseed(self):
        blocked = (self.canvas[BLUE] + self.canvas[BLACK]) > 0
        y, x = self.seed_cell
        if blocked[y, x]:
            free = np.argwhere(~blocked)
            if free.size == 0:
                return                                   # nowhere left -- skip, don't crash
            y, x = free[np.argmin(((free - [y, x]) ** 2).sum(1))]
        seed = torch.tensor([[int(y), int(x)]], device=self.dev)
        self.state = seed_state(Teacher(self.cfg, seed, self.masks).rgba(self.masks), self.cfg)

    def clear(self):
        self.canvas.fill(0)
        self.environment_dirty = True
        self.state, self.t = None, 0

    @torch.no_grad()
    def step(self):
        self.masks = torch.from_numpy(self.canvas)[None].to(self.dev)
        self.fields.relax(self.masks, self.cfg.init_iters if self.t == 0 else self.cfg.field_iters)
        if self.state is None or self.state[0, 3].sum().item() < 0.5:
            self.reseed()
        kill = torch.maximum(self.masks[:, BLUE:BLUE+1], self.masks[:, BLACK:BLACK+1])
        sense, boost = self.fields.sense(self.masks), self.masks[:, GREEN:GREEN+1]
        for _ in range(self.steps_per_frame):
            self.state = self.model(self.state, self._weights(), sense, kill, boost)
        self.t += 1
        return self.render()

    def render(self):
        P_ = self.panel_px
        world_view = self.environment_image().copy()
        colony = self.environment_image().copy()

        ## Match the environment's pixel language: each alive NCA cell becomes a
        ## small bacterium dot, with a brighter pulsating edge at the colony front.
        alpha = self.state[0, 3].detach().clamp(0, 1).cpu().numpy()
        alive = (alpha > self.cfg.alive_threshold).astype(np.uint8)
        inner = cv2.erode(alive, np.ones((3, 3), np.uint8), iterations=1)
        front = (alive > 0) & (inner == 0)
        density = cv2.filter2D(alive.astype(np.float32), -1, np.ones((3, 3), np.float32))
        ys, xs = np.nonzero(alive)
        scale = P_ / self.g
        for gy, gx in zip(ys, xs):
            phase = self.breathe_phase[gy, gx]
            speed = self.breathe_speed[gy, gx]
            wave = 0.5 + 0.32 * math.sin(speed * self.t + phase) \
                   + 0.18 * math.sin(0.47 * speed * self.t + 1.7 * phase)
            pulse = 0.20 + 0.80 * np.clip(wave, 0, 1)
            feeding = self.canvas[RED, gy, gx] > 0
            crowd = density[gy, gx]
            colour = (max(35, int(145 - 10 * crowd)), min(255, int(145 + 14 * crowd)),
                      max(30, int(105 - 7 * crowd))) if not feeding else (185, 105, 230)
            if front[gy, gx]:
                colour = tuple(min(255, int(v + 28)) for v in colour)
            r = max(2, int(0.28 * scale)) + (0 if feeding else int(2 * pulse))
            px, py = int((gx + 0.5) * scale), int((gy + 0.5) * scale)
            cv2.rectangle(colony, (px - r, py - r), (px + r, py + r), colour, -1)

        ## Keep environmental objects static and on top of the breathing cells.
        self.draw_environment_pixels(colony)

        for view, title, detail in ((world_view, "ENVIRONMENT", "paint the conditions"),
                                    (colony, "COLONY RESPONSE", f"nutrient {self.n:.2f}  |  step {self.t}")):
            cv2.rectangle(view, (0, 0), (P_, 42), (250, 250, 250), -1)
            cv2.line(view, (0, 42), (P_, 42), (224, 224, 224), 1)
            cv2.putText(view, title, (14, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        (44, 71, 59), 1, cv2.LINE_AA)
            cv2.putText(view, detail, (14, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                        (125, 138, 132), 1, cv2.LINE_AA)
        return np.concatenate([world_view, colony], axis=1)

def draw_sidebar(h, w, selected, brush):
    bar = np.full((h, w, 3), 250, np.uint8)
    row = TOOL_H
    for i, name in enumerate(TOOLS):
        y0 = i * row
        active = name == selected
        fill = (235, 247, 239) if active else (250, 250, 250)
        edge = (96, 160, 120) if active else (226, 226, 226)
        cv2.rectangle(bar, (8, y0 + 7), (w - 8, y0 + row - 7), fill, -1)
        cv2.rectangle(bar, (8, y0 + 7), (w - 8, y0 + row - 7), edge, 1)
        c = (90, 90, 90) if name == "erase" else tuple(int(v*255) for v in PERSON_RGB[COLOURS.index(name)][::-1])
        cv2.circle(bar, (27, y0 + row // 2), 11, c, -1, cv2.LINE_AA)
        cv2.putText(bar, TOOL_LABELS[name], (45, y0 + row // 2 + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.36, (54, 68, 61), 1, cv2.LINE_AA)
        if active:
            cv2.rectangle(bar, (8, y0 + 7), (w - 8, y0 + row - 7), (96, 160, 120), 2)

    y0, brush_h = row * len(TOOLS) + 8, BRUSH_H
    cv2.putText(bar, "BRUSH", (12, y0 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                (125, 138, 132), 1, cv2.LINE_AA)
    for i, (label, radius) in enumerate((("PIXEL", 0), ("MED", 2), ("BOLD", 4))):
        top = y0 + i * brush_h
        active = radius == brush
        fill = (235, 247, 239) if active else (250, 250, 250)
        edge = (96, 160, 120) if active else (226, 226, 226)
        cv2.rectangle(bar, (8, top + 3), (w - 8, top + brush_h - 3), fill, -1)
        cv2.rectangle(bar, (8, top + 3), (w - 8, top + brush_h - 3), edge, 2 if active else 1)
        cv2.circle(bar, (27, top + brush_h // 2), radius + 3, (70, 70, 70), -1, cv2.LINE_AA)
        cv2.putText(bar, label, (45, top + brush_h // 2 + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.36, (54, 68, 61), 1, cv2.LINE_AA)
    reset_top = y0 + 3 * brush_h + 8
    button_h = (h - reset_top - 8) // 2
    cv2.rectangle(bar, (8, reset_top), (w - 8, reset_top + button_h - 3), (238, 242, 240), -1)
    cv2.rectangle(bar, (8, reset_top), (w - 8, reset_top + button_h - 3), (188, 202, 193), 1)
    cv2.putText(bar, "RESET", (36, reset_top + 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (54, 86, 67), 1, cv2.LINE_AA)
    clear_top = reset_top + button_h + 3
    cv2.rectangle(bar, (8, clear_top), (w - 8, h - 8), (246, 240, 240), -1)
    cv2.rectangle(bar, (8, clear_top), (w - 8, h - 8), (210, 188, 188), 1)
    cv2.putText(bar, "CLEAR", (36, clear_top + 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (110, 70, 70), 1, cv2.LINE_AA)
    return bar

def main():
    model, cfg = load_checkpoint(os.path.join("outputs", "bacteria_live.pth"))   # path to your saved checkpoint
    sess = VirtualSession(model, cfg, grid=cfg.grid)
    tool, brush = ["red"], [0]
    window, sidebar_w = "Biofilm simulation", 120
    cv2.namedWindow(window)
    drag = {"on": False, "prev": None}

    def on_mouse(ev, x, y, flags, _):
        P_ = sess.panel_px
        if x < sidebar_w:
            if ev == cv2.EVENT_LBUTTONDOWN:
                tool_h = TOOL_H
                brush_top = tool_h * len(TOOLS) + 8
                reset_top = brush_top + 3 * BRUSH_H + 8
                button_h = (P_ - reset_top - 8) // 2
                if y < tool_h * len(TOOLS):
                    tool[0] = TOOLS[min(y // tool_h, len(TOOLS) - 1)]
                elif brush_top <= y < brush_top + 3 * BRUSH_H:
                    brush[0] = (0, 2, 4)[min((y - brush_top) // BRUSH_H, 2)]
                elif reset_top <= y < reset_top + button_h:
                    sess.reseed()
                elif y >= reset_top + button_h:
                    sess.clear()
            return
        gx, gy = x - sidebar_w, y
        if not (0 <= gx < P_ and 42 <= gy < P_):
            return
        cell = (int(gy * sess.g / P_), int(gx * sess.g / P_))
        if ev == cv2.EVENT_LBUTTONDOWN:
            drag["on"], drag["prev"] = True, cell
            sess.paint(tool[0], *cell, brush[0], erase=(tool[0] == "erase"))
        elif ev == cv2.EVENT_LBUTTONUP:
            drag["on"], drag["prev"] = False, None
        elif ev == cv2.EVENT_MOUSEMOVE and drag["on"]:
            sess.paint(tool[0], *cell, brush[0], erase=(tool[0] == "erase"))
            drag["prev"] = cell

    cv2.setMouseCallback(window, on_mouse)
    while True:
        panel = sess.step()
        bar = draw_sidebar(panel.shape[0], sidebar_w, tool[0], brush[0])
        cv2.imshow(window, np.concatenate([bar, panel], axis=1))
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("r"):
            sess.reseed()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
