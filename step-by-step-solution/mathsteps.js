const mathsteps = require('mathsteps');
const problems = [
    "((5 + -3x) - (-5)) = -1x",
    "((-1) + 4x) = (-9x + (-10))",
    "((-7) + -6x) = (-3x + 5)",
    "9x = 3",
    "(-2x + 9) = (-10x + (-3))",
    "((-4) + 3x) = (10x + (-6))",
    "(-6) = (3x + -2x)",
    "(-8) = ((-5) + ((-6) / x))",
    "(7x - 6) = (3x - (-3))",
    "4x = (-6x + (-5))",
    "(8 - (-5)) = ((2x + (-7)) - 1)",
    "(-4) = (((-7) / x) + 8)",
    "(3x / 2) = (10 / 8)",
    "(-3x + 6) = (-1x + 2)",
    "10 = ((-2) + ((-4) / x))",
    "-7x = (-5)",
    "((7 - 6x) - 4) = (((-8) + 8x) - (-7))",
    "5x = ((-7x - 2) - (-4))",
    "((-6) + -9x) = (-2x + 1)"
]

let all_solutions = []
problems.forEach(problem => {

    let preprossedP =problem.replace(/()/g, '');
    const steps = mathsteps.solveEquation(preprossedP);

    let solution = steps.map(step=>step.newEquation.ascii())
    all_solutions.push({"problem": problem, "solution":solution})
})


const fs = require('fs')
const jsonString = JSON.stringify(all_solutions)
fs.writeFile('math_step_solutions.json', jsonString, err => {
    if (err) {
        console.log('Error writing file', err)
    } else {
        console.log('Successfully wrote file')
    }
})

