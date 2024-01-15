import { guiShader } from "./guiShader.js";
import { computeShader } from "./computeShader.js";

export default class Shader {
    constructor() { };
    static gui = guiShader;
    static compute = computeShader;
}