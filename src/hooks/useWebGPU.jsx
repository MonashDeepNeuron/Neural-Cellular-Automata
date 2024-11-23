import { useEffect, useRef } from "react";
import BufferManager from "../managers/BufferManager";
import { guiShader } from "../shaders/guiShader";
import { computeShader } from "../shaders/computeShader";
import { parseRuleString } from '../util/Parse'

const SQUARE_VERTICIES = new Float32Array([
    // X,    Y,
    -0.8, -0.8, // Triangle 1
    -0.8, 0.8,
    0.8, 0.8,

    0.8, 0.8, // Triangle 2
    0.8, -0.8,
    -0.8, -0.8,
]);

export default function useWebGPU(canvasRef) {
    const deviceRef = useRef(null);
    const contextRef = useRef(null);

    useEffect(() => {
        const initializeWebGPU = async () => {
            if (!canvasRef.current) return;

            // Check if WebGPU is supported
            if (!navigator.gpu) {
                console.error("WebGPU not supported on this browser.");
                return;
            }

            try {
                // Request GPU adapter and device
                const adapter = await navigator.gpu.requestAdapter();
                if (!adapter) {
                    console.error("No suitable GPUAdapter found.");
                    return;
                }
                const device = await adapter.requestDevice();
                deviceRef.current = device;

                // Configure the canvas context
                const canvas = canvasRef.current;
                const context = canvas.getContext("webgpu");


                // DOING FIRST RENDER PASS FOR TEST AND TO CLEAR 
                const canvasFormat = navigator.gpu.getPreferredCanvasFormat();

                context.configure({
                    device,
                    format: canvasFormat,
                });
                contextRef.current = context;

                // Perform an initial render pass to clear the canvas
                const encoder = device.createCommandEncoder();
                const pass = encoder.beginRenderPass({
                    colorAttachments: [
                        {
                            view: context.getCurrentTexture().createView(),
                            loadOp: "clear",
                            clearValue: { r: 0, g: 0, b: 0, a: 1 }, // Black background
                            storeOp: "store",
                        },
                    ],
                });
                pass.end();

                // Submit commands to the GPU
                device.queue.submit([encoder.finish()]);
                console.log("WebGPU initialized and canvas cleared.");
            } catch (error) {
                console.error("Failed to initialize WebGPU:", error);
            }
        };

        initializeWebGPU();

        return () => {
            // Cleanup (if needed)
            deviceRef.current = null;
            contextRef.current = null;
        };
    }, [canvasRef]);

    return { device: deviceRef.current, context: contextRef.current };
};
