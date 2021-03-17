const _ = require('lodash');
const config = require('../../config.json');

export default async (req, res) => {
  const r = await fetch(config['curriculumServer'] + '/post-test')
  const json = await r.json();
  res.json(json);
};
