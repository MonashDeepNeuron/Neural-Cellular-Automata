export default class DeviceManager {
    static canvas = document.querySelector("canvas");
    static device;

    static async getAdapter() {
        const adapter = await navigator.gpu.requestAdapter();
        return await adapter.requestDevice();
    }

    static async staticConstructor() {
        DeviceManager.device = await DeviceManager.getAdapter()
    }
}

