export default class shaderManager {
    static vertexShader = {
        label: "vertex shader",
        code:
            /*wgsl*/`
            @vertex
            fn vertexMain(@location(0) pos: vec2f) -> @builtin(position) vec4f{
                return vec4f(pos, 0, 1);
            }
            `
    }
    static fragmentShader = {
        label: "fragment shader",
        code:
            /*wgsl*/`
            @fragment
            fn fragmentMain() -> @location(0) vec4f {
                return vec4f(1, 0, 0, 1); // RGBA
            }
            `
    }
}