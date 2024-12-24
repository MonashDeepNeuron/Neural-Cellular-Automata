import { useEffect, useState } from "react";
import { incrementStep } from "../store/webGPUSlice";
import { useDispatch, useSelector } from "react-redux";
import { createRenderPass, createComputePass } from "../util/ShaderPass"

export const useRenderLoop = ({ settings, resources }) => {
    const { workgroupSize, gridSize } = settings;
    const { running, framesPerUpdateLoop, step, template } = useSelector((state) => state.webGPU);

    const dispatch = useDispatch();


    useEffect(() => {

        if (!resources) {
            console.log("WebGPU initialization incomplete.");
            return;
        }

        const { device, context, pipelines, buffers } = resources;

        /**
        * Defining renderPass and computePass
        */
        const renderPass = createRenderPass(context, pipelines, buffers, gridSize);
        const computePass = createComputePass(pipelines, buffers, gridSize, workgroupSize);

        let intervalID;


        /**
        * Defining callback renderLoop
        */
        const renderLoop = () => {
            const encoder = device.createCommandEncoder();

            for (let i = 0; i < framesPerUpdateLoop; i++) {
                computePass(encoder, step);
                dispatch(incrementStep());
            }

            renderPass(encoder, step);
            device.queue.submit([encoder.finish()]);

        };

        if (running) {
            intervalID = setInterval(renderLoop, 50)
        }

        return () => {
            clearInterval(intervalID);
        };

    }, [running, step, template]);
};
