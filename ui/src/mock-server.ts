

const express = require('express');
const cors = require('cors');

const app = express();
const port = 3001;

app.use(cors());
app.use(express.json());

app.post('/api/generateDocumentation', (req: { body: { urls: any; types: any; }; }, res: { json: (arg0: { type: string; url: string; }[]) => void; }) => {
  const { urls, types } = req.body;
  const docs: { type: string, url: string }[] = [];
  urls.forEach((url: string, i: number) => {
    types.forEach((type: string) => {
      docs.push({
        type,
        url: `${url}/generated-${type}-doc-${i + 1}`
      });
    });
  });
  setTimeout(() => res.json(docs), 1000); // Simulate network delay
});

app.listen(port, () => {
  console.log(`Mock server running at http://localhost:${port}`);
});