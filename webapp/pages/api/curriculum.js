const _ = require('lodash');
const { UserSession } = require('../../lib/data');

const config = require('../../config.json');

export default async (req, res) => {
  const params = JSON.parse(req.query.params || '{}');
  const sessionId = params.id;
  const session = await UserSession.findOne({ id: sessionId });

  const next = await fetch(config['curriculumServer'] + '/next',
                           {
                             'method': 'POST',
                             'headers': { 'Content-Type': 'application/json' },
                             'body': JSON.stringify({
                               'curriculum': session.curriculum,
                               'student_history': session.exerciseResponses,
                             }),
                           });

  const json = await next.json();
  res.json(json);
};
