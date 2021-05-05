const config = require('./config.json');

export const TUTOR_URL = config['tutor-server-url'];

export function tutorRequestUrl(route, params) {
  return TUTOR_URL + '/' + route + '?params=' + encodeURIComponent(JSON.stringify(params));
}

export async function tutorRequest(route, params) {
  const response = await fetch(tutorRequestUrl(route, params));
  const jsonResponse = await response.json();
  return jsonResponse;
}
