import os
import cv2
import numpy as np
import torch
from bacteria_core import (Config, BacteriaNCA, seed_state, Fields, Teacher,
                           composite, load_checkpoint, COLOURS, PERSON_RGB,
                           RED, ORANGE, BLACK, BLUE, GREEN)

TOOLS = ["red", "orange", "black", "blue", "green", "erase"]

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

    def _weights(self):
        if self._wn != self.n:
            self._w, self._wn = self.model.weights_for(torch.tensor([self.n], device=self.dev)), self.n
        return self._w

    def paint(self, name, gy, gx, r, erase=False):
        d = disc_mask(self.g, gy, gx, r)
        self.canvas[:, d] = 0
        if not erase:
            self.canvas[COLOURS.index(name), d] = 1

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
        world_view = composite(torch.zeros(4, self.g, self.g), self.masks[0])
        colony = composite(self.state[0, :4], self.masks[0])
        to_bgr = lambda im: cv2.resize(np.ascontiguousarray((np.clip(im, 0, 1)[..., ::-1]*255).astype(np.uint8)),
                                       (P_, P_), interpolation=cv2.INTER_NEAREST)
        return np.concatenate([to_bgr(world_view), to_bgr(colony)], axis=1)

def draw_sidebar(h, w, selected):
    bar = np.full((h, w, 3), 230, np.uint8)
    row = h // len(TOOLS)
    for i, name in enumerate(TOOLS):
        y0 = i * row
        c = (90, 90, 90) if name == "erase" else tuple(int(v*255) for v in PERSON_RGB[COLOURS.index(name)][::-1])
        cv2.rectangle(bar, (6, y0+6), (w-6, y0+row-6), c, -1)
        if name == selected:
            cv2.rectangle(bar, (2, y0+2), (w-2, y0+row-2), (0, 0, 0), 2)
    return bar

def main():
    model, cfg = load_checkpoint(os.path.join("outputs", "bacteria.pth"))   # path to your saved checkpoint
    sess = VirtualSession(model, cfg, grid=cfg.grid)
    tool, brush = ["red"], 2
    window, sidebar_w = "virtual demo", 70
    cv2.namedWindow(window)
    drag = {"on": False, "prev": None}

    def on_mouse(ev, x, y, flags, _):
        P_ = sess.panel_px
        if x < sidebar_w:
            if ev == cv2.EVENT_LBUTTONDOWN:
                tool[0] = TOOLS[min(y // (sidebar_w * len(TOOLS) // sidebar_w), len(TOOLS)-1)]
            return
        gx, gy = x - sidebar_w, y
        if not (0 <= gx < P_ and 0 <= gy < P_):
            return
        cell = (int(gy * sess.g / P_), int(gx * sess.g / P_))
        if ev == cv2.EVENT_LBUTTONDOWN:
            drag["on"], drag["prev"] = True, cell
        elif ev == cv2.EVENT_LBUTTONUP:
            drag["on"], drag["prev"] = False, None
        elif ev == cv2.EVENT_MOUSEMOVE and drag["on"]:
            sess.paint(tool[0], *cell, brush, erase=(tool[0] == "erase"))
            drag["prev"] = cell

    cv2.setMouseCallback(window, on_mouse)
    while True:
        panel = sess.step()
        bar = draw_sidebar(panel.shape[0], sidebar_w, tool[0])
        cv2.imshow(window, np.concatenate([bar, panel], axis=1))
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("r"):
            sess.reseed()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()