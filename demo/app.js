// Demo web application for movie recommender
const express = require('express');
const fetch = require('node-fetch');
const fs = require('fs');
const path = require('path');
const app = express();

const PORT = 8080;
const MOVIES_METADATA_PATH = path.join(__dirname, '../data/ml-20m-psm/movies_metadata.csv');
const POSTERS_PATH = path.join(__dirname, '../data/ml-20m-psm/posters/');
const USER_GROUPS_PATH = path.join(__dirname, 'user_groups.json');

let moviesMetadata = {}; // To store metadata indexed by movie ID
let userGroups = { lessthan50: [], from51to150: [], from151to300: [], morethan300: [], all: [] }; // To store user groups

// Load movie metadata from CSV
fs.createReadStream(MOVIES_METADATA_PATH)
  .pipe(require('csv-parser')())
  .on('data', (row) => {
    moviesMetadata[row.id] = row;
  })
  .on('end', () => {
    console.log('Movie metadata loaded.');
  });

// Load preprocessed user groups
if (fs.existsSync(USER_GROUPS_PATH)) {
  userGroups = JSON.parse(fs.readFileSync(USER_GROUPS_PATH));
  console.log('User groups loaded from file:', {
    lessthan50: userGroups.lessthan50.length,
    from51to150: userGroups.from51to150.length,
    from151to300: userGroups.from151to300.length,
    morethan300: userGroups.morethan300.length,
    all: userGroups.all.length
  });
} else {
  console.error('User groups file not found. Please run the preprocessing script.');
}

// Serve static files (like posters and HTML)
app.use('/posters', express.static(POSTERS_PATH));
app.use(express.static(path.join(__dirname, 'public')));

// Serve the demo page
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// API to get random user IDs for buttons
app.get('/get-random-users', (req, res) => {
  const getRandomFromList = (list) => list[Math.floor(Math.random() * list.length)];

  res.json({
    single: getRandomFromList(userGroups.lessthan50),
    ten: getRandomFromList(userGroups.from51to150),
    fifty: getRandomFromList(userGroups.from151to300),
    hundred: getRandomFromList(userGroups.morethan300),
    random: getRandomFromList(userGroups.all)
  });
});

app.listen(PORT, () => {
  console.log(`Demo web application running at http://127.0.0.1:${PORT}/`);
});
