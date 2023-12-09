export default class bindGroupManager {
    static layoutDescriptor = {
        label: "bind group layout",
        entries: [{
            binding: 0,
            visibility: GPUShaderStage.VERTEX,
            buffer: { type: "read-only-storage" },
        }],
    }

    static createDescriptor({
        layout: bindGroupLayout,
        buffer: vertexBuffer,
    }) {
        return {
            label: "basic bind group thing",
            layout: bindGroupLayout,
            entries: [{
                binding: 0,
                resource: { buffer: vertexBuffer },
            }],
        }
    }
}