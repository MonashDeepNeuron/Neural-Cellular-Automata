export default class pipelineManager {
    static createLayoutDescriptor({
        bindGroupLayout: bindGroupLayout,
    }) {
        return {
            label: "pipeline layout",
            bindGroupLayouts: [bindGroupLayout],
        }
    }

    static createDescriptor({
        shaders: shaders,
        vertexBufferLayout: vertexBufferLayout,
        format: format,
        layout: pipelineLayout,
    }) {

        return {
            label: "main pipeline",
            vertex: {
                module: shaders.vertex,
                entryPoint: "vertexMain",
                buffers: [vertexBufferLayout],
            },
            fragment: {
                module: shaders.fragment,
                entryPoint: "fragmentMain",
                targets: [{ format: format }],
            },
            primitive: { topology: "triangle-list" },
            layout: pipelineLayout,
        }
    }
}