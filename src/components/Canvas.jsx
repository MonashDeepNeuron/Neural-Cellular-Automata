import React, { useRef, useState, useEffect } from "react";
import useWebGPU from "../hooks/useWebGPU";
import startingPatterns from "../patterns/startingPatterns";
import useWebGPUResources from "../hooks/useGPUResources";

// Circular bugs initial state 
const INITIAL_STATE = startingPatterns[8];
const SETTINGS = {
    WORKGROUP_SIZE: 16,
    INITIAL_TEMPLATE_NO: 9,
    INITIAL_STATE: INITIAL_STATE,
    GRID_SIZE: startingPatterns[8].minGrid,
    ruleString: INITIAL_STATE.rule,
};

const Canvas = () => {
    /**
     * These are refs internal to the useWebGPU hook
     * so if WebGPUCanvas re-renders, these will not be recalculated
     */
    const [webGPUState, setWebGPUState] = useState({
        device: null,
        context: null,
    });

    const canvasRef = useRef(null);
    const { device, context } = useWebGPU(canvasRef, SETTINGS);

    useEffect(() => {
        if (device && context) {
            setWebGPUState({ device, context });
        }
    }, [device, context]);

    // Log the state for debugging
    useEffect(() => {
        if (webGPUState.device && webGPUState.context) {
            console.log("WebGPU state updated:", webGPUState);
        }
    }, [webGPUState]);


    return (
        <canvas
            ref={canvasRef}
            width={SETTINGS.GRID_SIZE}
            height={SETTINGS.GRID_SIZE}
            style={{ width: "100%", height: "100%", display: "block" }}
        />
    );
};

export default Canvas;
