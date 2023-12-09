// textures, buffers, bindings

export const vertices = new Float32Array([
    // X,    Y,
    -0.8, -0.8,
    -0.8, 0.8,
    0.8, 0.8,
]);

export const vertexBufferLayout = {
    arrayStride: 8,
    attributes: [{
        format: "float32x2",
        offset: 0,
        shaderLocation: 0,
    }]
};

export function vertexBufferDescriptor(vertices) {
    return {
        label: "vertex buffer",
        size: vertices.byteLength,
        usage:
            GPUBufferUsage.VERTEX |
            GPUBufferUsage.COPY_DST |
            GPUBufferUsage.STORAGE,
    };
};



