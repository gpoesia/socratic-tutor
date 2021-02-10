const { UserSession } = require('../../lib/data');
const uuid = require('uuid');
const _ = require('lodash');
const seedrandom = require('seedrandom');

const Config = require('../../config.json');

export default async (req, res) => {
  const params = JSON.parse(req.query.params || '{}');
  const sessionId = params.id;

  if (!sessionId) {
    return res.json({ "error": "No session ID provided." });
  }

  const session = await UserSession.findOne({ id: sessionId });

  const rng = seedrandom(session.id);
  const originalOrder = (_.range(Config.testProblems.length)
                          .map(() => rng()));
  const testProblemsOrder = _.clone(originalOrder);
  testProblemsOrder.sort();
  const testProblemsPermutation = testProblemsOrder.map(
                                    (v) => originalOrder.indexOf(v));

  // Pick the first problem not seen of each difficulty.
  const seenDifficulties = new Set();
  const problems = [];

  const TestProblemsByID = {};
  Config.testProblems.forEach(p => { TestProblemsByID[p.id] = p; });

  for (let i = 0; i < testProblemsOrder.length; i++) {
    const p = Config.testProblems[testProblemsPermutation[i]];

    if (!seenDifficulties.has(p.difficulty) &&
        _.find(session.preTestResponses,
               (r) => TestProblemsByID[r.id].difficulty == p.difficulty) === undefined) {
      seenDifficulties.add(p.difficulty);
      problems.push({
        id: p.id,
        problem: p.problem,
        variable: p.variable,
      });
    }
  }

  res.send(problems);
};
