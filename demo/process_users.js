const fs = require('fs');
const path = require('path');
const csv = require('csv-parser');

const RATINGS_PATH = path.join(__dirname, '../data/ml-20m-psm/ratings.csv');
const USER_GROUPS_PATH = path.join(__dirname, 'user_groups.json');

let userGroups = { single: [], ten: [], fifty: [], hundred: [], all: [] };

// Process ratings to categorize users by number of ratings
fs.createReadStream(RATINGS_PATH)
  .pipe(csv({ separator: ',', headers: ['userId', 'movieId', 'rating', 'timestamp'] }))
  .on('data', (row) => {
    if (!userGroups.all.includes(row.userId)) {
      userGroups.all.push(row.userId);
    }
  })
  .on('end', () => {
    const userRatingsCount = {};

    // Calculate ratings count per user
    userGroups.all.forEach((userId) => {
      userRatingsCount[userId] = (userRatingsCount[userId] || 0) + 1;
    });

    // Categorize users based on their ratings count
    Object.entries(userRatingsCount).forEach(([userId, count]) => {
      if (count === 1) userGroups.single.push(userId);
      else if (count >= 5 && count <= 15) userGroups.ten.push(userId);
      else if (count >= 35 && count <= 65) userGroups.fifty.push(userId);
      else if (count >= 60 && count <= 140) userGroups.hundred.push(userId);
    });

    // Persist user groups to file
    fs.writeFileSync(USER_GROUPS_PATH, JSON.stringify(userGroups, null, 2));
    console.log('User groups saved to file:', USER_GROUPS_PATH);
  });
