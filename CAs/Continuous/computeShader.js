
const ACTIVATION_FUNCTION_FLAG = '//INSERT ACTIVATION BETWEEN FLAGS$$$$$';


/** Compute shader run every time the update loop runs
 * Computes grid updates  
 * (DO NOT REMOVE wgsl comment. This is for the WGSL Literal extension)
 * @todo Replace this and ComputeShaderManager.getComputeShaderWithActivation with one function. 
 * This would de-necessitate the flag thing
 */ 
const computeShader =
    /*wgsl*/`
    @group(0) @binding(0) var<uniform> grid: vec2f;
    @group(0) @binding(1) var<storage> cellStateIn: array<f32>;
    @group(0) @binding(2) var<storage, read_write> cellStateOut: array<f32>;
    @group(0) @binding(3) var<storage> rule: array<f32>;
    override WORKGROUP_SIZE: u32 = 16;
    
    fn cellIndex(cell: vec2u) -> u32 {
        // Supports grid wrap-around
        // grid.x and grid.y 
        return (cell.y % u32(grid.y))*u32(grid.x)+(cell.x % u32(grid.x));
    }

    fn cellValue(x : u32, y: u32) -> f32 {
        return cellStateIn[cellIndex(vec2(x,y))];
    }

    // This part of the shader is replacable
    fn applyActivation(x : f32) -> f32 {
        ${ACTIVATION_FUNCTION_FLAG}
        // This part here is replaceable, please do not edit out 
        // as this is a flag to find this section to replace easily
        // See function insertActivation()
        return -1./pow(2., (0.6*pow(x, 2.)))+1.;
        ${ACTIVATION_FUNCTION_FLAG}
    }


    @compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE)
    fn computeMain(@builtin(global_invocation_id) cell: vec3u) {

        let i = cellIndex(cell.xy);

        // Perform Convolution. We will use step function as activation function for now
        // k1 | k2 | k3
        // k4 | k5 | k6
        // k7 | k8 | k9
        let k1 = cellValue(cell.x-1, cell.y-1) * rule[0];
        let k2 = cellValue(cell.x, cell.y-1) * rule[1] ;
        let k3 = cellValue(cell.x+1, cell.y-1) * rule[2] ;
        let k4 = cellValue(cell.x-1, cell.y) * rule[3] ;
        let k5 = cellValue(cell.x, cell.y) * rule[4] ;
        let k6 = cellValue(cell.x+1, cell.y) * rule[5] ;
        let k7 = cellValue(cell.x-1, cell.y+1) * rule[6] ;
        let k8 = cellValue(cell.x, cell.y+1) * rule[7] ;
        let k9 = cellValue(cell.x+1, cell.y+1) * rule[8] ;

        let result = k1 + k2 + k3 + k4 + k5 + k6 + k7 + k8 + k9 ;

        cellStateOut[i] = applyActivation(result);
                
    }`;





    export class ComputeShaderManager {

        static currentComputeShaderText = "";
        static currentActivation = "return x;";
        static simulationPipeline = null;
        static simulationShaderModule = null;
        static newActivation = false;
        static workgroupSize = 16; // only 1, 2, 4, 8, 16 work. higher is smoother. // There is a limitation though to some pcs/graphics cards
        static pipelineLayout = {}


        static setWorkgroupSize(newSize){
            this.workgroupSize = newSize;
        }

        static setPipelineLayout(newLayout){
            this.pipelineLayout = newLayout;
        }

        static getComputeShaderWithActivation (activationFunctionString){
            let shaderTemplate = computeShader.split(ACTIVATION_FUNCTION_FLAG);
    
            console.log(shaderTemplate);
    
            let newShader = shaderTemplate[0] + '\n' 
                + ACTIVATION_FUNCTION_FLAG + '\n'
                + activationFunctionString + '\n'
                + ACTIVATION_FUNCTION_FLAG + '\n'
                + shaderTemplate[2] ;
            console.log(`NEW SHADER \n ${newShader}`)
            return newShader;
        }

        static updateActivationField(activationText){
            document.getElementById("activationTextField").value = activationText;
        }
        
        static getActivationFieldText(){
            return document.getElementById("activationTextField").value;
        }

        static submitActivation(){
            ComputeShaderManager.newActivation = true;
            const newActivaiton = ComputeShaderManager.getActivationFieldText();
            ComputeShaderManager.updateActivationField(newActivaiton);
            const newComputeShader = ComputeShaderManager.getComputeShaderWithActivation(newActivaiton);
            ComputeShaderManager.currentActivation = newActivaiton;
            ComputeShaderManager.currentComputeShaderText = newComputeShader;
            console.log(`CURRENT ACTIVATION: ${ComputeShaderManager.currentActivation}`);
        }
        

        static compileNewSimulationPipeline(device){
            // load shader module for running simulation
            this.simulationShaderModule = device.createShaderModule({
                label: 'shader that computes next state',
                code: ComputeShaderManager.getComputeShaderWithActivation(this.currentActivation),
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
            
            return { simulationModule:this.simulationShaderModule, simulationPipeline:this.simulationPipeline};

        }

        /**
         * Initial setup for the compute shader including pipelines 
         */
        static initialSetup(device){
            
            document.getElementById("submitActivation").addEventListener("click", this.submitActivation);
            this.submitActivation();
            this.compileNewSimulationPipeline(device);
            this.newActivaiton = false;
        }
    }