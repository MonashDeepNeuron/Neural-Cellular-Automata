// this is supposed to be shader code

function isNear(origin, target, neighbourhoodType, radius) {
    x_offset = abs(origin.x - target.x);
    y_offset = abs(origin.y - target.y)
    switch (neighbourhoodType) {
        case 'vonNeumann':
            distance = x_offset + y_offset
        case 'moore':
            distance = max(x_offset, y_offset)
        case 'circular':
            distance = sqrt(x_offset ** 2 + y_offset ** 2)
    }
    return distance < radius
}

// from this, create array of relative neighbours, then write it to buffer.
/*
wg id --> cell
local id --> displacement. offset necessary

one more storage thing --> array for active neighbours


CS1

each workgroup goes for one cell
each thread is for one neighbour

CS2
map active neighbours to future state

so then, two compute pipelines
countComputePipeline
updateComputePipeline


*/