// parse rule string for life like automata

function parseRuleString(ruleString, rule) {
    // ruleString is given by the user. it is a string
    // rule is in the buffer, we update it to match ruleString
    // rule is a U32 integer array length 1
    rule[0] = 0; // reset the rule
    slashFlag = false
    for (let i = 0; i < ruleString.length; i++) {
        char = ruleString[i];
        if (char === "/") {
            slashFlag = !slashFlag
            continue
        }
        let num = Number(char) // the character is indeed a number
        switch (slashFlag) {
            case false: // before "/" sign. survival case
                //console.log(rule[0] + (2 ** (num + 9)))
                rule[0] += 2 ** (num + 9)
                break;
            case true: // after "/" sign. birth case
                rule[0] += 2 ** num
        }
    }
}