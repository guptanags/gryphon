const fs = require('fs');
const xml2js = require('xml2js');
const { execSync } = require('child_process');
const xmllint = require('xmllint');

// Read the XML file
const filePath = './gryphon-ai-architecture.xml';
const xmlContent = fs.readFileSync(filePath, 'utf-8');

// Parse the XML
const parser = new xml2js.Parser();
parser.parseString(xmlContent, (err, result) => {
  if (err) {
    console.error('Error parsing XML:', err);
    return;
  }

  const ids = new Set();
  const missingParents = [];
  const missingSources = [];
  const missingTargets = [];
  // Define a list of known supported elements and attributes
  const supportedElements = ['mxCell', 'mxGeometry']; // Add more as needed
  const supportedAttributes = ['id', 'parent', 'source', 'target', 'value', 'style']; // Add more as needed

  const unsupportedElements = [];
  const unsupportedAttributes = [];

  // Collect all IDs
  const cells = result.mxfile.diagram[0].mxGraphModel[0].root[0].mxCell;
  const idMap = new Map();

  cells.forEach((cell) => {
    const id = cell.$.id;
    if (ids.has(id)) {
      console.error(`Duplicate ID found: ${id}`);
    } else {
      ids.add(id);
      idMap.set(id, cell);
    }

    // Check parent references
    if (cell.$.parent && !ids.has(cell.$.parent)) {
      missingParents.push(cell.$.parent);
    }

    // Check source/target references for edges
    if (cell.$.source && !ids.has(cell.$.source)) {
      missingSources.push(cell.$.source);
    }
    if (cell.$.target && !ids.has(cell.$.target)) {
      missingTargets.push(cell.$.target);
    }

    // Correctly extract element names
    const elementName = cell['#name'] || cell.$?.name || 'unknown'; // Fallback to 'unknown' if name is not found
    const attributes = Object.keys(cell.$ || {});

    // Check for unsupported elements
    if (!supportedElements.includes(elementName)) {
      unsupportedElements.push(elementName);
    }

    // Check for unsupported attributes
    attributes.forEach((attr) => {
      if (!supportedAttributes.includes(attr)) {
        unsupportedAttributes.push(attr);
      }
    });
  });

  // Report issues
  if (missingParents.length > 0) {
    console.error('Missing parent references:', missingParents);
  }
  if (missingSources.length > 0) {
    console.error('Missing source references:', missingSources);
  }
  if (missingTargets.length > 0) {
    console.error('Missing target references:', missingTargets);
  }
  if (unsupportedElements.length > 0) {
    console.error('Unsupported Elements:', unsupportedElements);
  } else {
    console.log('No unsupported elements found.');
  }

  if (unsupportedAttributes.length > 0) {
    console.error('Unsupported Attributes:', unsupportedAttributes);
  } else {
    console.log('No unsupported attributes found.');
  }

  if (missingParents.length === 0 && missingSources.length === 0 && missingTargets.length === 0) {
    console.log('No issues found in the XML.');
  }
});

// Check file encoding
try {
  const encoding = execSync(`file -I ${filePath}`).toString();
  console.log('File Encoding:', encoding);
} catch (error) {
  console.error('Error checking file encoding:', error.message);
}

// Validate XML structure
try {
  const validationResult = xmllint.validateXML({
    xml: xmlContent,
    schema: null, // Provide schema if available
  });

  if (validationResult.errors) {
    console.error('XML Validation Errors:', validationResult.errors);
  } else {
    console.log('XML is well-formed and valid.');
  }
} catch (error) {
  console.error('Error during XML validation:', error.message);
}
