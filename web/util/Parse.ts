export const NEIGHBOURHOOD_TYPES: Record<string, number> = {
	M: 0, // Moore/square
	N: 1, // Neumann/diamond/radius measured by Manhatten distance
	C: 2 // Centre of cell within sqrt(dx^2 + dy^2)
};

/** Convenience function, probably exists in a library*/
function isDigit(c: string) {
	return c >= '0' && c <= '9';
}



/**
 * Interprets the rulestring according to the following notation for Larger than Life CA.
 * Rulestring is given as Rr, Cc, Ss Bb, Nn where:
 *  - r is the radius in units
 *  - c is how many cell states there are.
 *  - n is the type of neighbourhood (eg. moore, neumann, cicular)
 *  - s is ranges in the form a-b, c-d, e, ... conditions of survival
 *  - b is ranges in the form a-b, c-d, e, ... conditions of birth
 *
 * For cell states, all cells are born on maximum cell state, then decrease one point for
 * each cycle they are not within the survival bounds (so they don't die instantly)
 * Cell states greater than 0 are considered alive still.
 *
 * See https://conwaylife.com/wiki/Larger_than_Life
 * @param ruleString
 * @returns rule array in format: [r, c, no. srange, su, sl, su, sl, ... , no. brange, bu, bl, bu, bl, ... , n]
 * @todo this function returns null with invalid input, needs to be accounted for in other code
 * @todo BUG - When no given birth or survival cases given, results in error
 * @todo rulestring validation function - pretty much accounted for in current interface setup though
 */
export function parseLTLRule(raw: string): Uint32Array | null {
	const ruleString = raw.replace(/\s/g, '').toUpperCase(); // Cut out white space and assert uppercase

	// console.log("Getting RULE...");
	// ruleString is given by the user. it is a string
	// Output format
	// [r, c, n, no. srange, sl, su, sl, su, ... , no. brange, bl, bu, bl, bu, ...]
	// For the sake of efficiency, for S and B,
	// if the number is specifically just the number, rather than a range, this
	// is denoted through the duplication of the number
	// eg. 2-5, 7, 11-13, ...,
	// push 2, 5, 7, 7, 11, 13, ...,
	if (ruleString.length < 8) {
		return null;
	}

	const ruleList: number[] = [];

	let i = 0;
	if (ruleString[i] !== 'R') {
		// console.log("ERROR");
		return null;
	}

	const nextNumber = () => {
		//console.log(`Parsing at ${i}`);
		while (!isDigit(ruleString[i])) {
			i++;
		}

		const numStart = i;
		while (isDigit(ruleString[i])) {
			i++;
		}
		//console.log(`Extracted ${ruleString.substring(numStart, i)}`)
		return Number(ruleString.substring(numStart, i));
	};

	i++;

	ruleList.push(nextNumber()); // Radius
	ruleList.push(nextNumber()); // C (multiple states)
	if (ruleList[1] < 2) {
		ruleList[1] = 2; // Lowest possiple number of states allowable
	}
	ruleList.push(0); // Save a space for neighbourhood type

	// Parse survival cases

	while (ruleString[i] !== 'S') {
		i++;
	}
	ruleList.push(0);
	const sConditionCountIndex = ruleList.length - 1;

	// Rule list should be 4 long now
	// See how many characters are occupies by survival rule
	let lastS = i; // Index of the start of last set of numbers
	// 2-4, 34-36, B3-4, ...
	//    ^ -----

	while (ruleString[lastS] !== 'B') {
		lastS++;
	}
	lastS -= 3;
	while (ruleString[lastS] !== ',' && ruleString[lastS] !== 'S') {
		lastS--;
	}

	// console.log(`Last S range at index ${lastS}, current index at ${i}`);

	// eg. 2-5, 7, 11-13, ...,
	// push 2, 5, 7, 7, 11, 13, ...,
	while (i <= lastS) {
		ruleList.push(nextNumber());
		ruleList[sConditionCountIndex]++;

		while (ruleString[i] !== '-' && ruleString[i] !== ',') {
			//console.log(`Rejected ${ruleString[i]} in survive parse`);
			i++;
		}

		if (ruleString[i] === '-') {
			i++; // Exclude dash to prevent misinterpretation as negative
			ruleList.push(nextNumber());
			ruleList[sConditionCountIndex]++;
		} else {
			ruleList.push(ruleList[ruleList.length - 1]);
			ruleList[sConditionCountIndex]++;
		}
	} // NOTE: this will over-push by one number i.e. it will include the
	// number referred to by lastS

	//console.log(`Update after finding survivials: ${ruleList}`);

	ruleList.push(0);
	const bConditionCountIndex = ruleList.length - 1;

	// Rule list should be 3 long now
	// See how many characters are occupies by survival rule
	let lastB = i; // Index of the start of last set of numbers
	// 2-4, 34-36, B3-4, ...
	//    ^ -----

	while (ruleString[lastB] !== 'N') {
		lastB++;
	}
	lastB -= 3;
	while (ruleString[lastB] !== ',' && ruleString[lastB] !== 'B') {
		lastB--;
	}

	//console.log(`Last B range at index ${lastB}`);

	// eg. 2-5, 7, 11-13, ...,
	// push 2, 5, 7, 7, 11, 13, ...,
	while (i <= lastB) {
		ruleList.push(nextNumber());
		ruleList[bConditionCountIndex]++;

		while (ruleString[i] !== '-' && ruleString[i] !== ',') {
			i++;
		}

		if (ruleString[i] === '-') {
			ruleList.push(nextNumber());
			ruleList[bConditionCountIndex]++;
		} else {
			ruleList.push(ruleList[ruleList.length - 1]);
			ruleList[bConditionCountIndex]++;
		}
	} // NOTE: this may over-push by one number i.e. it will include the
	// number refered to by lastB

	// from i: ,Nn, S..., B...,
	// Assume n is a single character

	while (ruleString[i] !== 'N') {
		i++;
	}
	i++;

	ruleList[2] = NEIGHBOURHOOD_TYPES[ruleString[i]];

	i++;

	const RULE = new Uint32Array(ruleList.length);
	for (let i = 0; i < ruleList.length; i++) {
		RULE[i] = ruleList[i];
	}

	// console.log(ruleString);
	// console.log(ruleList);
	return RULE;
}



/**
 * Interprets the rulestring according to the following notation for Life-Like CA.
 * Rulestring is given as survival/birth where:
 *  - survival are all the numbers 0-9 where cells survive
 *  - birth are all the numbers 0-9 where cells are born
 *
 * For cell states, all cells are either alive or dead
 *
 * See https://conwaylife.com/wiki/Life-like
 * @param ruleString
 * @returns rule array in format: [r, c, no. srange, su, sl, su, sl, ... , no. brange, bu, bl, bu, bl, ... , n], since Life-like is a subset of Larger than life
 * @todo this function returns null with invalid input, needs to be accounted for in other code
 * @todo BUG - When no given birth or survival cases given, results in error
 * @todo rulestring validation function - pretty much accounted for in current interface setup though
 */
export function parseLifeLikeRule(raw: string): Uint32Array | null {
	const ruleString = raw.replace(/\s/g, '').toUpperCase(); // Cut out white space and assert uppercase

	// console.log("Getting RULE...");
	// ruleString is given by the user. it is a string
	// Output format
	// [r, c, n, no. srange, sl, su, sl, su, ... , no. brange, bl, bu, bl, bu, ...]
	// For S and B if the number is specifically just the number, rather than a range, this
	// is denoted through the duplication of the number
	// eg. 2-5, 7, 11-13, ...,
	// push 2, 5, 7, 7, 11, 13, ...,
	if (ruleString.length < 1) {
		return null;
	}

	// Radius r is fixed at 1, 
	// Number of cell states c is fixed at 2 
	// Neighbourhood type is fixed at Moore
	const ruleList: number[] = [1, 0, 0];

	// Parse survival cases
	let i = 0;

	// Parse survival cases
	ruleList.push(0);
	const sConditionCountIndex = ruleList.length - 1;
	while (ruleString[i] !== '/') {
		console.log(ruleString[i])
		ruleList.push(Number(ruleString[i]));
		ruleList.push(Number(ruleString[i]));
		// Re-push the previous number to account for the fact that
		// the number is a single number, not a range
		ruleList[sConditionCountIndex] += 2;
		i++;
	}

	i++; // Skip the slash

	// Parse survival cases
	ruleList.push(0);
	const bConditionCountIndex = ruleList.length - 1;
	while (i < ruleString.length) {
		console.log(ruleString[i])
		ruleList.push(Number(ruleString[i]));
		ruleList.push(Number(ruleString[i]));
		// Re-push the previous number to account for the fact that
		// the number is a single number, not a range
		ruleList[bConditionCountIndex] += 2;
		i++;
	}

	const RULE = new Uint32Array(ruleList.length);
	for (let i = 0; i < ruleList.length; i++) {
		RULE[i] = ruleList[i];
	}

	console.log(`Life-like: ${RULE}`);

	return RULE;
}


/**
 * Defines how to display the rulestring (which is formatted according to the
 * expected format for Larger than Life CA) in the Larger than Life specific interface.
 *
 * Rulestring is given as Rr, Cc, Ss Bb, Nn where:
 *  - r is the radius in units
 *  - c is how many cell states there are.
 *  - s is ranges in the form a-b, c-d, e, ... conditions of survival
 *  - b is ranges in the form a-b, c-d, e, ... conditions of birth
 *  - n is the type of neighbourhood (eg. moore, neumann, cicular)
 *
 * @param {String} ruleString
 */
export function displayLTLRule(ruleString: string) {
	// console.log(`Displaying ${ruleString}`);

	let i = 0;
	while (ruleString[i] !== 'R') {
		i++;
	}
	i++;

	let R = '';

	while (ruleString[i] !== ',') {
		R += ruleString[i];
		i++;
	}

	const inputR = document.getElementById('simulationInputR');
	if (inputR) (inputR as HTMLInputElement).value = R;

	while (ruleString[i] !== 'C') {
		i++;
	}
	i++;

	let C = '';

	while (ruleString[i] !== ',') {
		C += ruleString[i];
		i++;
	}

	const inputC = document.getElementById('simulationInputC');
	if (inputC) (inputC as HTMLInputElement).value = C;

	while (ruleString[i] !== 'S') {
		i++;
	}
	i++;

	const sStart = i;

	while (ruleString[i] !== 'B') {
		i++;
	}

	while (ruleString[i] !== ',') {
		i--;
	}

	const S = ruleString.substring(sStart, i);
	const inputS = document.getElementById('simulationInputS');
	if (inputS) (inputS as HTMLInputElement).value = S;

	while (ruleString[i] !== 'B') {
		i++;
	}
	i++;

	const bStart = i;

	while (ruleString[i] !== 'N') {
		i++;
	}

	while (ruleString[i] !== ',') {
		i--;
	}

	const B = ruleString.substring(bStart, i);
	const inputB = document.getElementById('simulationInputB');
	if (inputB) (inputB as HTMLInputElement).value = B;

	while (ruleString[i] !== 'N') {
		i++;
	}
	i++;

	const inputN = document.getElementById('simulationInputN');
	if (inputN) (inputN as HTMLInputElement).value = ruleString[i];
}