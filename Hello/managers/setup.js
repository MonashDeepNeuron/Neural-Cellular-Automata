export default class setupManager {
    static async device() {
        const adapter = await navigator.gpu.requestAdapter();
        return await adapter.requestDevice();
    };

    static context({
        device: device,
        format: format,
    }) {
        const canvas = document.querySelector("canvas");
        const context = canvas.getContext("webgpu");
        context.configure({
            device: device,
            format: format,
        })
        return context
    }
}