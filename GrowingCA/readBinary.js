const fs = require('fs').promises;
/*
Takes in binary encoding of model's weights in the form :
128 

# TODO: EXPORT AS FLOAT NOT INTEGERS
*/
async function loadBinaryFileAsIntegers(filePath) {
    try {
        const fileBuffer = await fs.readFile(filePath);
        const intArray = new Int32Array(fileBuffer.buffer, fileBuffer.byteOffset, fileBuffer.byteLength / Int32Array.BYTES_PER_ELEMENT);

        return intArray;
    } catch (error) {
        console.error('Failed to load file:', error);
        throw error;
    }
}


loadBinaryFileAsIntegers('../model_weights_pls_god.bin').then(integers => {
    console.log(integers);
}).catch(error => {
    console.error('Error:', error);
});
