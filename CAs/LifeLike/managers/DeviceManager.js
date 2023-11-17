export default class DeviceManager {
    static canvas = document.querySelector("canvas");
    static device;

    static async deviceSetup() {
        const adapter = await navigator.gpu.requestAdapter();
        return await adapter.requestDevice();
    }

    static async staticConstructor() {
        DeviceManager.device = DeviceManager.deviceSetup();
    }
}
