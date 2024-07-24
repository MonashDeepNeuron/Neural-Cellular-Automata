
const ACTIVATION_FUNCTION_FLAG = '//INSERT ACTIVATION BETWEEN FLAGS';

import { computeShaderCode } from "./computeShaderCode.js";


export class ComputeShaderManager {
    static simulationPipeline = null;
    static simulationShaderModule = null;
    static workgroupSize = 2; // only 1, 2, 4, 8, 16 work. higher is smoother. // There is a limitation though to some pcs/graphics cards // TOD: set to 16 and centralise definition
    static pipelineLayout = {}


    static setWorkgroupSize(newSize) {
        this.workgroupSize = newSize;
    }

    static setPipelineLayout(newLayout) {
        this.pipelineLayout = newLayout;
    }

    static compileNewSimulationPipeline(device) {
        // load shader module for running simulation
        this.simulationShaderModule = device.createShaderModule({
            label: 'shader that computes next state',
            code: computeShaderCode,
            constants: { WORKGROUP_SIZE: this.workgroupSize }
        }
        );
        // SIMULATION PIPELINE
        this.simulationPipeline = device.createComputePipeline({
            label: "Simulation pipeline",
            layout: this.pipelineLayout,
            compute: {
                module: this.simulationShaderModule,
                entryPoint: "computeMain",
            }
        });

        return { simulationModule: this.simulationShaderModule, simulationPipeline: this.simulationPipeline };

    }

    /**
     * Initial setup for the compute shader including pipelines 
     */
    static initialSetup(device) {
        this.compileNewSimulationPipeline(device);
    }
}