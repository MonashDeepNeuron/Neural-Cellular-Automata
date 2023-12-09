export default class bufferManager {
    static triangle = new Float32Array([
        // X,    Y,
        -0.8, -0.8,
        -0.8, 0.8,
        0.8, 0.8,
    ]);

    static square = new Float32Array([
        // X,    Y,
        -0.8, -0.8,
        -0.8, 0.8,
        0.8, 0.8,

        // X,    Y,
        -0.8, -0.8,
        0.8, -0.8,
        0.8, 0.8,
    ]);

    static layout = {
        arrayStride: 8,
        attributes: [{
            format: "float32x2",
            offset: 0,
            shaderLocation: 0,
        }]
    };

    static createDescriptor({ size: size }) {
        return {
            label: "vertex buffer",
            size: size,
            usage:
                GPUBufferUsage.VERTEX |
                GPUBufferUsage.COPY_DST |
                GPUBufferUsage.STORAGE,
        };
    };
}
