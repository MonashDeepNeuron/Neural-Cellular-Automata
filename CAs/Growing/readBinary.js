export { loadBinaryFileAsIntegers };
const fs = require('fs').promises;

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


