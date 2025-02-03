export default class DeviceManager {
	static canvas = document.querySelector('canvas');

	/** GPUDevice object */
	static device;

	static async getAdapter() {
		const adapter = await navigator.gpu.requestAdapter();
		return await adapter.requestDevice();
	}

	/**
	 * Gets access to the GPU device
	 */
	static async staticConstructor() {
		DeviceManager.device = await DeviceManager.getAdapter();
	}
}
