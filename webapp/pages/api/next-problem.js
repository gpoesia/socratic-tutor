const lodash = require('lodash');
const SOLUTION_GRAPH = require('../../solution_graph.json');

SOLUTION_GRAPH.forEach((p, i) => p.id = i);

const config = require('../../config.json');

const problemLevel = p => Math.floor(p.solution.length / config.levelRange);

const MINIMUM_LEVEL = lodash.min(SOLUTION_GRAPH.map(problemLevel));

export default (req, res) => {
  const params = JSON.parse(req.query.params || '{}');
  const policy = params.policy || 'random';
  const lastProblem = (params.lastProblem ? SOLUTION_GRAPH[params.lastProblem] : null);
  const mistakes = params.mistakes || [];
  const targetLevel = lastProblem ? problemLevel(lastProblem) + 1 : MINIMUM_LEVEL;
  const nextLevelProblems = SOLUTION_GRAPH.filter(p => (problemLevel(p) == targetLevel));
  const curriculum = (policy === 'curriculum' ||
                      policy === 'personalized_curriculum');

  let chosen = null;

  if (policy === 'random') {
    chosen = lodash.sample(SOLUTION_GRAPH);
  }

  // In any of the curriculum cases, if not chosen yet, sample problem from next level.
  if (curriculum && !chosen && nextLevelProblems.length) {
    chosen = lodash.sample(nextLevelProblems);
  }

  if (chosen) {
    res.json({
      id: chosen['id'],
      solution: chosen['solution'],
      negatives: chosen['negative-examples'],
    });
  } else {
    res.json({
      id: null,
      solution: [],
      negatives: [],
    });
  }
};
