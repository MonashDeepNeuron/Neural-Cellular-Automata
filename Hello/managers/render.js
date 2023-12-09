
export default class renderManager {
    static createDescriptor({
        texture: texture,
    }) {
        return {
            label: "pass encoder",
            colorAttachments: [{
                view: texture.createView(), // to view the texture
                loadOp: "clear", // better for performance than load.
                clearValue: { r: 0, g: 0, b: 0.4, a: 1.0 }, // what is empty
                storeOp: "store", // try discard later
            }],
        }
    }
}