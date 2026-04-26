import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import axios from "axios";
import cron from "node-cron";
import multer from "multer";
import fs from "fs";
import path from "path";
import { randomUUID } from "crypto";
import { fileURLToPath } from "url";
import { GoogleGenerativeAI } from "@google/generative-ai";

dotenv.config();

const app = express();

app.use(cors());
app.use(express.json({ limit: "50mb" }));
app.use(express.urlencoded({ extended: true, limit: "50mb" }));

const PORT = process.env.PORT || 3000;

const GRAPH_HOST = process.env.GRAPH_HOST || "https://graph.instagram.com";
const GRAPH_VERSION = process.env.GRAPH_VERSION || "v25.0";

const META_ACCESS_TOKEN = process.env.META_ACCESS_TOKEN;
const IG_USER_ID = process.env.IG_USER_ID;
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

const GEMINI_TEXT_MODEL = process.env.GEMINI_TEXT_MODEL || "gemini-1.5-flash";
const GEMINI_IMAGE_MODEL =
  process.env.GEMINI_IMAGE_MODEL || "gemini-2.5-flash-image-preview";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const uploadsDir = path.join(__dirname, "uploads");

if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir, { recursive: true });
}

app.use("/uploads", express.static(uploadsDir));

const upload = multer({
  dest: uploadsDir,
  limits: {
    fileSize: 25 * 1024 * 1024
  }
});

/**
 * Temporary memory storage.
 * Good for testing only.
 * On Render restart/redeploy, scheduled posts will be lost.
 * Later use Supabase/PostgreSQL.
 */
const posts = [];

app.use((req, res, next) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.originalUrl}`);
  next();
});

function getBaseUrl(req) {
  const protocol = req.headers["x-forwarded-proto"] || req.protocol;
  return `${protocol}://${req.get("host")}`;
}

function requireMetaConfig() {
  if (!META_ACCESS_TOKEN || !IG_USER_ID) {
    throw new Error(
      "Missing META_ACCESS_TOKEN or IG_USER_ID in environment variables."
    );
  }
}

function requireGeminiConfig() {
  if (!GEMINI_API_KEY) {
    throw new Error("Missing GEMINI_API_KEY in environment variables.");
  }
}

function isPublicUrl(url) {
  try {
    const parsed = new URL(url);
    return parsed.protocol === "https:" || parsed.protocol === "http:";
  } catch {
    return false;
  }
}

function normalizeHashtags(hashtags) {
  if (!hashtags) return [];

  if (Array.isArray(hashtags)) {
    return hashtags
      .map((tag) => String(tag).trim())
      .filter(Boolean)
      .map((tag) => (tag.startsWith("#") ? tag : `#${tag}`));
  }

  if (typeof hashtags === "string") {
    return hashtags
      .split(/[\s,]+/)
      .map((tag) => tag.trim())
      .filter(Boolean)
      .map((tag) => (tag.startsWith("#") ? tag : `#${tag}`));
  }

  return [];
}

function getBodyValue(body, camelKey, snakeKey) {
  return body[camelKey] ?? body[snakeKey];
}

function parseJsonLoose(text) {
  try {
    return JSON.parse(String(text).replace(/```json|```/g, "").trim());
  } catch {
    return null;
  }
}

async function fetchImageAsInlineData(imageUrl) {
  const response = await axios.get(imageUrl, {
    responseType: "arraybuffer",
    timeout: 30000
  });

  const mimeType =
    response.headers["content-type"] &&
    response.headers["content-type"].startsWith("image/")
      ? response.headers["content-type"]
      : "image/jpeg";

  return {
    inlineData: {
      mimeType,
      data: Buffer.from(response.data).toString("base64")
    }
  };
}

function saveBase64Image({ base64Data, mimeType, req }) {
  let ext = ".png";

  if (mimeType === "image/jpeg" || mimeType === "image/jpg") ext = ".jpg";
  if (mimeType === "image/webp") ext = ".webp";

  const fileName = `${randomUUID()}${ext}`;
  const filePath = path.join(uploadsDir, fileName);

  fs.writeFileSync(filePath, Buffer.from(base64Data, "base64"));

  return `${getBaseUrl(req)}/uploads/${fileName}`;
}

/**
 * Instagram publish:
 * 1. POST /{IG_USER_ID}/media
 * 2. POST /{IG_USER_ID}/media_publish
 */
async function publishToInstagram({ imageUrl, caption }) {
  requireMetaConfig();

  if (!imageUrl || !isPublicUrl(imageUrl)) {
    throw new Error("imageUrl must be a public direct URL.");
  }

  const createContainerUrl = `${GRAPH_HOST}/${GRAPH_VERSION}/${IG_USER_ID}/media`;

  const containerResponse = await axios.post(createContainerUrl, null, {
    params: {
      image_url: imageUrl,
      caption: caption || "",
      access_token: META_ACCESS_TOKEN
    }
  });

  const creationId = containerResponse.data?.id;

  if (!creationId) {
    throw new Error("Meta did not return creation_id.");
  }

  const publishUrl = `${GRAPH_HOST}/${GRAPH_VERSION}/${IG_USER_ID}/media_publish`;

  const publishResponse = await axios.post(publishUrl, null, {
    params: {
      creation_id: creationId,
      access_token: META_ACCESS_TOKEN
    }
  });

  return {
    creationId,
    publishId: publishResponse.data?.id
  };
}

async function generateTextWithGemini(prompt, modelName = GEMINI_TEXT_MODEL) {
  requireGeminiConfig();

  const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
  const model = genAI.getGenerativeModel({ model: modelName });

  const result = await model.generateContent(prompt);
  return result.response.text();
}

async function generateTextWithGeminiVision({
  imageUrl,
  prompt,
  modelName = GEMINI_TEXT_MODEL
}) {
  requireGeminiConfig();

  const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
  const model = genAI.getGenerativeModel({ model: modelName });

  try {
    const imagePart = await fetchImageAsInlineData(imageUrl);
    const result = await model.generateContent([prompt, imagePart]);
    return result.response.text();
  } catch (error) {
    console.warn("Gemini vision fallback to text-only:", error.message);
    const result = await model.generateContent(`${prompt}\n\nImage URL: ${imageUrl}`);
    return result.response.text();
  }
}

async function generateCaptionWithGemini({
  imageUrl,
  language = "arabic",
  tone = "premium",
  model = GEMINI_TEXT_MODEL
}) {
  const prompt = `
You are a social media marketing assistant for artificial trees, artificial flowers, and luxury decoration products.

Analyze the provided product image and generate Instagram content.

Language:
${language}

Tone:
${tone}

Return strict JSON only:
{
  "caption": "short marketing caption",
  "hashtags": ["#tag1", "#tag2", "#tag3", "#tag4", "#tag5", "#tag6", "#tag7", "#tag8"],
  "alt_text": "short alt text"
}
`;

  const text = await generateTextWithGeminiVision({
    imageUrl,
    prompt,
    modelName: model
  });

  const parsed = parseJsonLoose(text);

  if (parsed) {
    return {
      caption: parsed.caption || "",
      hashtags: normalizeHashtags(parsed.hashtags),
      alt_text: parsed.alt_text || ""
    };
  }

  return {
    caption: text,
    hashtags: [],
    alt_text: ""
  };
}

async function generateEditPromptWithGemini({
  imageUrl,
  editStyle = "luxury interior background",
  language = "english",
  model = GEMINI_TEXT_MODEL
}) {
  const prompt = `
Create a professional AI image editing prompt for this product image.

Edit style:
${editStyle}

Language:
${language}

The prompt must be for editing an existing product image, not generating from scratch.

Rules:
- Preserve the main product exactly.
- Keep the artificial tree, flowers, pot, planter, trunk, leaves, branches, shape, size, angle, and proportions exactly the same.
- Change only the background, environment, decoration, lighting, shadows, and composition.
- Make it photorealistic.
- Make it Instagram-ready.
- Default output size: 1080x1350.
- No cartoon, no painting, no AI-looking result.
- No text or watermark.

Return strict JSON only:
{
  "prompt": "full editing prompt here"
}
`;

  const text = await generateTextWithGeminiVision({
    imageUrl,
    prompt,
    modelName: model
  });

  const parsed = parseJsonLoose(text);

  return {
    prompt: parsed?.prompt || String(text).replace(/```json|```/g, "").trim()
  };
}

async function editImageWithGemini({
  originalImageUrl,
  prompt,
  model = GEMINI_IMAGE_MODEL,
  req
}) {
  requireGeminiConfig();

  if (!originalImageUrl || !isPublicUrl(originalImageUrl)) {
    throw new Error("originalImageUrl must be a public URL.");
  }

  if (!prompt) {
    throw new Error("prompt is required.");
  }

  const imagePart = await fetchImageAsInlineData(originalImageUrl);

  const requestBody = {
    contents: [
      {
        role: "user",
        parts: [
          {
            text: prompt
          },
          imagePart
        ]
      }
    ]
  };

  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`;

  const response = await axios.post(url, requestBody, {
    params: {
      key: GEMINI_API_KEY
    },
    headers: {
      "Content-Type": "application/json"
    },
    timeout: 120000
  });

  const parts = response.data?.candidates?.[0]?.content?.parts || [];

  const imageOutput = parts.find((part) => part.inlineData || part.inline_data);

  if (!imageOutput) {
    const textOutput = parts.map((part) => part.text).filter(Boolean).join("\n");

    throw new Error(
      textOutput ||
        "Gemini image model did not return image data. Check image model access."
    );
  }

  const inlineData = imageOutput.inlineData || imageOutput.inline_data;

  const editedImageUrl = saveBase64Image({
    base64Data: inlineData.data,
    mimeType: inlineData.mimeType || inlineData.mime_type || "image/png",
    req
  });

  return {
    edited_image_url: editedImageUrl,
    editedImageUrl,
    model
  };
}

/**
 * ROOT + HEALTH
 */
app.get("/", (req, res) => {
  res.json({
    ok: true,
    app: "AutoFlow Backend",
    status: "running",
    graphHost: GRAPH_HOST,
    graphVersion: GRAPH_VERSION
  });
});

app.get("/api/health", (req, res) => {
  res.json({
    ok: true,
    timestamp: new Date().toISOString(),
    graphHost: GRAPH_HOST,
    graphVersion: GRAPH_VERSION
  });
});

/**
 * META TEST
 */
async function testMetaConnection() {
  requireMetaConfig();

  const url = `${GRAPH_HOST}/${GRAPH_VERSION}/me`;

  const response = await axios.get(url, {
    params: {
      fields: "user_id,username",
      access_token: META_ACCESS_TOKEN
    }
  });

  return {
    account: response.data,
    configuredIgUserId: IG_USER_ID,
    idMatches: String(response.data.user_id) === String(IG_USER_ID),
    graphHost: GRAPH_HOST,
    graphVersion: GRAPH_VERSION
  };
}

app.get("/api/meta/test-connection", async (req, res) => {
  try {
    const result = await testMetaConnection();

    res.json({
      ok: true,
      ...result
    });
  } catch (error) {
    console.error("Meta test failed:", error.response?.data || error.message);
    res.status(500).json({
      ok: false,
      error: error.response?.data || error.message
    });
  }
});

app.post("/api/meta/test-connection", async (req, res) => {
  try {
    const result = await testMetaConnection();

    res.json({
      ok: true,
      ...result
    });
  } catch (error) {
    console.error("Meta test failed:", error.response?.data || error.message);
    res.status(500).json({
      ok: false,
      error: error.response?.data || error.message
    });
  }
});

/**
 * PUBLISH NOW
 */
app.post("/api/meta/publish-now", async (req, res) => {
  try {
    const imageUrl = getBodyValue(req.body, "imageUrl", "image_url");
    const caption = req.body.caption || req.body.final_text || req.body.finalText || "";

    const result = await publishToInstagram({
      imageUrl,
      caption
    });

    res.json({
      ok: true,
      result
    });
  } catch (error) {
    console.error("Publish now failed:", error.response?.data || error.message);
    res.status(500).json({
      ok: false,
      error: error.response?.data || error.message
    });
  }
});

/**
 * GEMINI TEST
 */
app.post("/api/gemini/test", async (req, res) => {
  try {
    requireGeminiConfig();

    const text = await generateTextWithGemini(
      'Return strict JSON only: {"ok": true, "message": "Gemini connected"}',
      req.body.model || GEMINI_TEXT_MODEL
    );

    res.json({
      ok: true,
      raw: text,
      result: parseJsonLoose(text) || { message: text }
    });
  } catch (error) {
    console.error("Gemini test failed:", error.message);
    res.status(500).json({
      ok: false,
      error: error.message
    });
  }
});

app.get("/api/ai/models", (req, res) => {
  res.json({
    ok: true,
    textModels: [
      "gemini-1.5-flash",
      "gemini-2.0-flash",
      "gemini-2.5-flash"
    ],
    imageModels: [
      "gemini-2.5-flash-image-preview",
      "gemini-2.5-flash-image"
    ],
    defaultTextModel: GEMINI_TEXT_MODEL,
    defaultImageModel: GEMINI_IMAGE_MODEL
  });
});

/**
 * GENERATE CAPTION
 */
app.post("/api/gemini/generate-caption", async (req, res) => {
  try {
    const imageUrl = getBodyValue(req.body, "imageUrl", "image_url");
    const language = req.body.language || "arabic";
    const tone = req.body.tone || "premium";
    const model = req.body.model || req.body.selectedModel || GEMINI_TEXT_MODEL;

    if (!imageUrl) {
      return res.status(400).json({
        ok: false,
        error: "imageUrl is required."
      });
    }

    const result = await generateCaptionWithGemini({
      imageUrl,
      language,
      tone,
      model
    });

    res.json({
      ok: true,
      result
    });
  } catch (error) {
    console.error("Generate caption failed:", error.message);
    res.status(500).json({
      ok: false,
      error: error.message
    });
  }
});

/**
 * GENERATE EDIT PROMPT
 */
app.post("/api/gemini/generate-edit-prompt", async (req, res) => {
  try {
    const imageUrl = getBodyValue(req.body, "imageUrl", "image_url");
    const editStyle =
      req.body.editStyle || req.body.edit_style || "luxury interior background";
    const language = req.body.language || "english";
    const model = req.body.model || req.body.selectedModel || GEMINI_TEXT_MODEL;

    if (!imageUrl) {
      return res.status(400).json({
        ok: false,
        error: "imageUrl is required."
      });
    }

    const result = await generateEditPromptWithGemini({
      imageUrl,
      editStyle,
      language,
      model
    });

    res.json({
      ok: true,
      result
    });
  } catch (error) {
    console.error("Generate edit prompt failed:", error.message);
    res.status(500).json({
      ok: false,
      error: error.message
    });
  }
});

/**
 * AI IMAGE EDITING
 */
app.post("/api/ai/edit-image", async (req, res) => {
  try {
    const originalImageUrl =
      getBodyValue(req.body, "originalImageUrl", "original_image_url") ||
      getBodyValue(req.body, "imageUrl", "image_url");

    const prompt = req.body.prompt || req.body.ai_edit_prompt || "";
    const model = req.body.model || GEMINI_IMAGE_MODEL;

    if (!originalImageUrl) {
      return res.status(400).json({
        ok: false,
        error: "originalImageUrl is required."
      });
    }

    if (!prompt) {
      return res.status(400).json({
        ok: false,
        error: "prompt is required."
      });
    }

    const result = await editImageWithGemini({
      originalImageUrl,
      prompt,
      model,
      req
    });

    res.json({
      ok: true,
      result
    });
  } catch (error) {
    console.error("AI edit image failed:", error.response?.data || error.message);
    res.status(500).json({
      ok: false,
      error: error.response?.data || error.message
    });
  }
});

/**
 * UPLOAD IMAGE
 */
app.post("/api/upload", upload.any(), (req, res) => {
  try {
    const file = req.files?.[0];

    if (!file) {
      return res.status(400).json({
        ok: false,
        error: "No image file uploaded. Send multipart/form-data with any file field."
      });
    }

    const ext = path.extname(file.originalname || "") || ".jpg";
    const newFileName = `${file.filename}${ext}`;
    const oldPath = file.path;
    const newPath = path.join(uploadsDir, newFileName);

    fs.renameSync(oldPath, newPath);

    const publicUrl = `${getBaseUrl(req)}/uploads/${newFileName}`;

    res.json({
      ok: true,
      url: publicUrl,
      imageUrl: publicUrl,
      image_url: publicUrl
    });
  } catch (error) {
    console.error("Upload failed:", error.message);
    res.status(500).json({
      ok: false,
      error: error.message
    });
  }
});

/**
 * CREATE SCHEDULED POST
 */
app.post("/api/posts", (req, res) => {
  const imageUrl = getBodyValue(req.body, "imageUrl", "image_url");
  const caption = req.body.caption || "";
  const hashtags = normalizeHashtags(req.body.hashtags);
  const scheduledAt = getBodyValue(req.body, "scheduledAt", "scheduled_at");

  if (!imageUrl || !scheduledAt) {
    return res.status(400).json({
      ok: false,
      error: "imageUrl and scheduledAt are required."
    });
  }

  const now = new Date().toISOString();
  const finalText = `${caption}\n${hashtags.join(" ")}`.trim();

  const post = {
    id: randomUUID(),

    imageUrl,
    image_url: imageUrl,

    caption,
    hashtags,

    finalText,
    final_text: finalText,

    scheduledAt,
    scheduled_at: scheduledAt,

    status: "approved",

    publishAttempts: 0,
    publish_attempts: 0,

    metaContainerId: null,
    meta_container_id: null,

    metaPublishId: null,
    meta_publish_id: null,

    errorMessage: null,
    error_message: null,

    createdAt: now,
    created_at: now,

    updatedAt: now,
    updated_at: now,

    publishedAt: null,
    published_at: null
  };

  posts.push(post);

  res.json({
    ok: true,
    post
  });
});

app.get("/api/posts", (req, res) => {
  res.json({
    ok: true,
    posts
  });
});

/**
 * RETRY FAILED POST
 */
app.post("/api/posts/:id/retry", async (req, res) => {
  const post = posts.find((p) => p.id === req.params.id);

  if (!post) {
    return res.status(404).json({
      ok: false,
      error: "Post not found."
    });
  }

  try {
    post.status = "publishing";
    post.publishAttempts += 1;
    post.publish_attempts = post.publishAttempts;
    post.updatedAt = new Date().toISOString();
    post.updated_at = post.updatedAt;

    const result = await publishToInstagram({
      imageUrl: post.imageUrl,
      caption: post.finalText
    });

    post.status = "published";
    post.metaContainerId = result.creationId;
    post.meta_container_id = result.creationId;
    post.metaPublishId = result.publishId;
    post.meta_publish_id = result.publishId;
    post.publishedAt = new Date().toISOString();
    post.published_at = post.publishedAt;
    post.updatedAt = new Date().toISOString();
    post.updated_at = post.updatedAt;
    post.errorMessage = null;
    post.error_message = null;

    res.json({
      ok: true,
      post
    });
  } catch (error) {
    post.status = "failed";
    post.errorMessage = JSON.stringify(error.response?.data || error.message);
    post.error_message = post.errorMessage;
    post.updatedAt = new Date().toISOString();
    post.updated_at = post.updatedAt;

    res.status(500).json({
      ok: false,
      post
    });
  }
});

/**
 * CRON SCHEDULER
 */
cron.schedule("*/5 * * * *", async () => {
  const now = new Date();

  const duePosts = posts.filter((post) => {
    return (
      post.status === "approved" &&
      new Date(post.scheduledAt) <= now &&
      post.publishAttempts < 3
    );
  });

  for (const post of duePosts) {
    try {
      post.status = "publishing";
      post.publishAttempts += 1;
      post.publish_attempts = post.publishAttempts;
      post.updatedAt = new Date().toISOString();
      post.updated_at = post.updatedAt;

      const result = await publishToInstagram({
        imageUrl: post.imageUrl,
        caption: post.finalText
      });

      post.status = "published";
      post.metaContainerId = result.creationId;
      post.meta_container_id = result.creationId;
      post.metaPublishId = result.publishId;
      post.meta_publish_id = result.publishId;
      post.publishedAt = new Date().toISOString();
      post.published_at = post.publishedAt;
      post.updatedAt = new Date().toISOString();
      post.updated_at = post.updatedAt;
      post.errorMessage = null;
      post.error_message = null;

      console.log(`Published post ${post.id}`);
    } catch (error) {
      post.status = "failed";
      post.errorMessage = JSON.stringify(error.response?.data || error.message);
      post.error_message = post.errorMessage;
      post.updatedAt = new Date().toISOString();
      post.updated_at = post.updatedAt;

      console.error(`Failed post ${post.id}:`, post.errorMessage);
    }
  }
});

/**
 * 404
 */
app.use((req, res) => {
  res.status(404).json({
    ok: false,
    error: `Endpoint not found: ${req.method} ${req.originalUrl}`
  });
});

app.listen(PORT, () => {
  console.log(`AutoFlow Backend running on port ${PORT}`);
});
