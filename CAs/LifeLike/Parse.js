import EventManager from "../Shared/managers/EventManager";

export function parseRuleString(ruleString) {
  // ruleString is given by the user. it is a string
  let RULE = new Uint32Array(1)
  let slashFlag = false // tells us whether we are before or after the / symbol
  for (let i = 0; i < ruleString.length; i++) {
      let char = ruleString[i];
      if (char === "/") {
          slashFlag = !slashFlag
          continue
      }
      let num = Number(char) // the character is indeed a number
      switch (slashFlag) {
          case false: // before "/" sign. survival case
              RULE[0] += 2 ** (num + 9)
              break;
          case true: // after "/" sign. birth case
              RULE[0] += 2 ** num
      }
  }
  return RULE
}


export function displayRule(ruleString){
    document.getElementById("simulationInput").value = ruleString;
}
