import { guiShader } from "./guiShader.js";
import { computeShader } from "./computeShader.js";

/**
 * Not sure this can be a shared resource for more complex models.
 * Must take into consideration that computeShader has different
 * implementation in each.
 * @todo Make the guiShader universal maybe? (probably through this class)
 */
export default class Shader {
    constructor() { };
    static gui = guiShader;
    static compute = computeShader;
}