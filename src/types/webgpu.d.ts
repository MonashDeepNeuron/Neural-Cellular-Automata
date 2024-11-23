// src/webgpu.d.ts
interface Navigator {
    gpu: GPU;
}

interface GPUCanvasContext extends CanvasRenderingContext {
    configure(options: {
        device: GPUDevice;
        format: GPUTextureFormat;
    }): void;
}
