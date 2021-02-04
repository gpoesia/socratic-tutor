const lodash = require('lodash');
const { problems, exercises } = require('../../solution_graph.json');

problems.forEach((p, i) => p.id = i);
exercises.forEach((e, i) => e.id = i);

const config = require('../../config.json');

const problemLevel = p => Math.floor(p.solution.length / config.levelRange);
const exerciseLevel = e => e['pos']['level'];

const MINIMUM_LEVEL = lodash.min(exercises.map(exerciseLevel));

export default (req, res) => {
  const params = JSON.parse(req.query.params || '{}');
  const policy = params.policy || 'random';
  const lastExercise = (params.lastExercise ? exercises[params.lastExercise] : null);
  const succeeded = params.succeeded || [];
  const targetLevel = lastExercise ? exerciseLevel(lastExercise) + 1 : MINIMUM_LEVEL;
  const nextLevelProblems = exercises.filter(e => (exerciseLevel(e) == targetLevel));
  const curriculum = (policy === 'curriculum' ||
                      policy === 'personalized_curriculum');

  let chosen = null;

  if (policy === 'random') {
    chosen = lodash.sample(exercises);
  }

  // In any of the curriculum cases, if not chosen yet, sample problem from next level.
  if (curriculum && !chosen && nextLevelProblems.length) {
    chosen = lodash.sample(nextLevelProblems);
  }

  if (chosen) {
    res.json(chosen);
  } else {
    res.json({
      id: null,
    });
  }
};
