export class bindGroup {

    static layoutDescriptor = {
        label: "bind group layout",
        entries: [{
            binding: 0,
            visibility: GPUShaderStage.VERTEX,
            buffer: { type: "read-only-storage" },
        }],
    };

    static descriptor = {
        label: "bind group",
        layout: null,
        entries: [{
            binding: 0,
            resource: { buffer: null },
        }],
    };
};

