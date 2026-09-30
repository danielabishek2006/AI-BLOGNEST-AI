const { GoogleGenerativeAI } = require('@google/generative-ai');

/**
 * Helper function to enforce a maximum timeout per API call.
 */
const fetchWithTimeout = (promise, ms = 10000) => {
  return Promise.race([
    promise,
    new Promise((_, reject) =>
      setTimeout(() => reject(new Error(`Request timed out after ${ms / 1000} seconds`)), ms)
    )
  ]);
};

const getModelsToTry = () => {
  const configuredModel = (process.env.GEMINI_MODEL || '').trim();
  const defaultModels = [
    'gemini-3.5-flash',
    'gemini-3.5-flash-lite',
    'gemini-3.8-flash',
  ];

  const models = [];
  const seen = new Set();

  for (const model of [configuredModel, ...defaultModels]) {
    if (!model) continue;
    const normalized = model.trim();
    if (!seen.has(normalized.toLowerCase())) {
      models.push(normalized);
      seen.add(normalized.toLowerCase());
    }
  }

  return models.length ? models : defaultModels;
};

/**
 * Executes a call to Gemini with model fallback, retries, and strict request timeouts.
 * 
 * @param {string} prompt - The text prompt.
 * @param {object} options - Optional parameters.
 * @param {number} options.timeoutMs - Maximum allowed duration per API attempt in ms (default: 8000ms).
 * @param {number} options.maxRetries - Retry attempts per model (default: 2).
 * @returns {Promise<string>} The generated text response.
 */
const callGemini = async (prompt, options = {}) => {
  const { timeoutMs = 8000, maxRetries = 3 } = options;
  const apiKey = process.env.GEMINI_API_KEY;

  if (!apiKey) {
    throw new Error('Gemini API key is missing from environment variables');
  }

  const modelsToTry = getModelsToTry();
  const genAI = new GoogleGenerativeAI(apiKey);
  let lastError = null;

  for (const modelName of modelsToTry) {
    let delay = 1500;

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        console.log(`[Gemini] Model '${modelName}' | Attempt ${attempt}/${maxRetries}`);

        const model = genAI.getGenerativeModel({ model: modelName });

        // Enforce maximum timeout on generateContent
        const result = await fetchWithTimeout(
          model.generateContent(prompt),
          timeoutMs
        );

        const response = result.response;
        const text = response.text();

        if (!text) {
          throw new Error('Empty text received from Gemini');
        }

        return text;
      } catch (error) {
        lastError = error;
        console.error(`[Gemini Error] ${modelName} (Attempt ${attempt}): ${error.message}`);

        if (
          error.message.includes('API key') ||
          error.message.includes('API_KEY_INVALID') ||
          error.message.includes('403') ||
          error.message.includes('Not Found')
        ) {
          break;
        }

        if (attempt < maxRetries) {
          await new Promise((res) => setTimeout(res, delay));
          delay *= 1.5;
        }
      }
    }

    console.warn(`[Gemini] Switching to next fallback model due to failures on '${modelName}'`);
  }

  throw new Error(`Gemini API service unavailable: ${lastError?.message || 'All models failed'}`);
};

module.exports = {
  callGemini,
};