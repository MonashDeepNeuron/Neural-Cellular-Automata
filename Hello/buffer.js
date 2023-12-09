// buffers

export class triangleBuffer {
    static vertices = new Float32Array([
        // X,    Y,
        -0.8, -0.8,
        -0.8, 0.8,
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
    static descriptor = {
        label: "vertex buffer",
        size: triangleBuffer.vertices.byteLength,
        usage:
            GPUBufferUsage.VERTEX |
            GPUBufferUsage.COPY_DST |
            GPUBufferUsage.STORAGE,
    };
}

export class squareBuffer {
    static vertices = new Float32Array([
        // X,    Y,
        -0.8, -0.8,
        -0.8, 0.8,
        0.8, 0.8,

        -0.8, -0.8,
        0.8, 0.8,
        0.8, -0.8,
    ]);

    static layout = {
        arrayStride: 8,
        attributes: [{
            format: "float32x2",
            offset: 0,
            shaderLocation: 0,
        }]
    };
    static descriptor = {
        label: "vertex buffer",
        size: squareBuffer.vertices.byteLength,
        usage:
            GPUBufferUsage.VERTEX |
            GPUBufferUsage.COPY_DST |
            GPUBufferUsage.STORAGE,
    };
}




