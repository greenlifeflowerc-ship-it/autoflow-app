import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import axios from "axios";
import cron from "node-cron";
import multer from "multer";
import fs from "fs";
import path from "path";
import sharp from "sharp";
import { v2 as cloudinary } from "cloudinary";
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

const GEMINI_TEXT_MODEL = process.env.GEMINI_TEXT_MODEL || "gemini-2.5-flash";
const GEMINI_IMAGE_MODEL =
  process.env.GEMINI_IMAGE_MODEL || "gemini-2.5-flash-image-preview";

const CLOUDINARY_CLOUD_NAME = process.env.CLOUDINARY_CLOUD_NAME;
const CLOUDINARY_API_KEY = process.env.CLOUDINARY_API_KEY;
const CLOUDINARY_API_SECRET = process.env.CLOUDINARY_API_SECRET;

cloudinary.config({
  cloud_name: CLOUDINARY_CLOUD_NAME,
  api_key: CLOUDINARY_API_KEY,
  api_secret: CLOUDINARY_API_SECRET,
  secure: true
});

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const tempDir = path.join(__dirname, "temp_uploads");

if (!fs.existsSync(tempDir)) {
  fs.mkdirSync(tempDir, { recursive: true });
}

const upload = multer({
  dest: tempDir,
  limits: {
    fileSize: 250 * 1024 * 1024
  }
});

// Temporary memory storage only.
// On Render restart/redeploy, scheduled posts are lost.
// Later replace with Supabase/PostgreSQL.
const posts = [];

app.use((req, res, next) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.originalUrl}`);
  next();
});

function requireMetaConfig() {
  if (!META_ACCESS_TOKEN || !IG_USER_ID) {
    throw new Error("Missing META_ACCESS_TOKEN or IG_USER_ID.");
  }
}

function requireGeminiConfig() {
  if (!GEMINI_API_KEY) {
    throw new Error("Missing GEMINI_API_KEY.");
  }
}

function requireCloudinaryConfig() {
  if (!CLOUDINARY_CLOUD_NAME || !CLOUDINARY_API_KEY || !CLOUDINARY_API_SECRET) {
    throw new Error(
      "Missing Cloudinary environment variables: CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET."
    );
  }
}

function isPublicHttpsUrl(url) {
  try {
    const parsed = new URL(url);
    return parsed.protocol === "https:";
  } catch {
    return false;
  }
}

function cleanupFile(filePath) {
  try {
    if (filePath && fs.existsSync(filePath)) {
      fs.unlinkSync(filePath);
    }
  } catch {
    // Ignore cleanup errors.
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

function detectMediaTypeFromUrl(url) {
  const lower = String(url || "").split("?")[0].toLowerCase();

  if (lower.endsWith(".jpg") || lower.endsWith(".jpeg") || lower.endsWith(".png") || lower.endsWith(".webp")) {
    return "image";
  }

  if (lower.endsWith(".mp4") || lower.endsWith(".mov") || lower.endsWith(".m4v")) {
    return "video";
  }

  return null;
}

function detectMediaTypeFromFile(file) {
  const mime = String(file.mimetype || "").toLowerCase();
  const name = String(file.originalname || "").toLowerCase();

  if (mime.startsWith("image/")) return "image";
  if (mime.startsWith("video/")) return "video";

  if (name.endsWith(".jpg") || name.endsWith(".jpeg") || name.endsWith(".png") || name.endsWith(".webp") || name.endsWith(".heic")) {
    return "image";
  }

  if (name.endsWith(".mp4") || name.endsWith(".mov") || name.endsWith(".m4v")) {
    return "video";
  }

  return null;
}

function normalizeMediaInput(body) {
  const rawMediaType =
    body.mediaType ||
    body.media_type ||
    null;

  const imageUrl =
    body.imageUrl ||
    body.image_url ||
    null;

  const videoUrl =
    body.videoUrl ||
    body.video_url ||
    null;

  const mediaUrl =
    body.mediaUrl ||
    body.media_url ||
    imageUrl ||
    videoUrl ||
    null;

  let mediaType = rawMediaType ? String(rawMediaType).toLowerCase() : null;

  if (!mediaType && imageUrl) mediaType = "image";
  if (!mediaType && videoUrl) mediaType = "video";
  if (!mediaType && mediaUrl) mediaType = detectMediaTypeFromUrl(mediaUrl);

  if (mediaType === "photo") mediaType = "image";
  if (mediaType === "reel") mediaType = "video";

  return {
    mediaUrl,
    imageUrl,
    videoUrl,
    mediaType
  };
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

function saveBase64TempImage({ base64Data, mimeType }) {
  let ext = ".png";

  if (mimeType === "image/jpeg" || mimeType === "image/jpg") ext = ".jpg";
  if (mimeType === "image/webp") ext = ".webp";

  const fileName = `${randomUUID()}${ext}`;
  const filePath = path.join(tempDir, fileName);

  fs.writeFileSync(filePath, Buffer.from(base64Data, "base64"));

  return filePath;
}

async function uploadImageToCloudinary(filePath) {
  requireCloudinaryConfig();

  const result = await cloudinary.uploader.upload(filePath, {
    resource_type: "image",
    folder: "autoflow/images",
    use_filename: false,
    unique_filename: true,
    overwrite: false,
    format: "jpg"
  });

  return result.secure_url;
}

async function uploadVideoToCloudinary(filePath) {
  requireCloudinaryConfig();

  const result = await cloudinary.uploader.upload(filePath, {
    resource_type: "video",
    folder: "autoflow/videos",
    use_filename: false,
    unique_filename: true,
    overwrite: false
  });

  return result.secure_url;
}

async function uploadNormalizedImage(file) {
  const normalizedPath = path.join(tempDir, `${randomUUID()}.jpg`);

  await sharp(file.path)
    .rotate()
    .jpeg({
      quality: 92,
      mozjpeg: true
    })
    .toFile(normalizedPath);

  const secureUrl = await uploadImageToCloudinary(normalizedPath);

  cleanupFile(file.path);
  cleanupFile(normalizedPath);

  return {
    mediaUrl: secureUrl,
    imageUrl: secureUrl,
    videoUrl: null,
    mediaType: "image",
    mimeType: "image/jpeg"
  };
}

async function uploadVideo(file) {
  const secureUrl = await uploadVideoToCloudinary(file.path);

  cleanupFile(file.path);

  return {
    mediaUrl: secureUrl,
    imageUrl: null,
    videoUrl: secureUrl,
    mediaType: "video",
    mimeType: file.mimetype || "video/mp4"
  };
}

async function pollMediaContainerStatus(containerId) {
  for (let attempt = 1; attempt <= 20; attempt += 1) {
    try {
      const response = await axios.get(`${GRAPH_HOST}/${GRAPH_VERSION}/${containerId}`, {
        params: {
          fields: "status_code,status",
          access_token: META_ACCESS_TOKEN
        },
        timeout: 30000
      });

      const statusCode = response.data?.status_code;
      const status = response.data?.status;

      if (statusCode === "FINISHED") {
        return response.data;
      }

      if (statusCode === "ERROR") {
        throw new Error(`Meta media container failed: ${status || "Unknown error"}`);
      }

      await new Promise((resolve) => setTimeout(resolve, 5000));
    } catch (error) {
      if (attempt === 20) throw error;
      await new Promise((resolve) => setTimeout(resolve, 5000));
    }
  }

  throw new Error("Timed out waiting for Meta media container processing.");
}

/**
 * Instagram publish:
 * image => image_url + media_type IMAGE
 * video => video_url + media_type VIDEO
 */
async function publishToInstagram({ mediaUrl, imageUrl, videoUrl, mediaType, caption }) {
  requireMetaConfig();

  const finalMediaUrl = mediaUrl || imageUrl || videoUrl;
  const finalMediaType = mediaType || detectMediaTypeFromUrl(finalMediaUrl);

  if (!finalMediaUrl || !isPublicHttpsUrl(finalMediaUrl)) {
    throw new Error("mediaUrl must be a public HTTPS URL.");
  }

  if (finalMediaType !== "image" && finalMediaType !== "video") {
    throw new Error("mediaType must be image or video.");
  }

  const createContainerUrl = `${GRAPH_HOST}/${GRAPH_VERSION}/${IG_USER_ID}/media`;

  const params = {
    caption: caption || "",
    access_token: META_ACCESS_TOKEN
  };

  if (finalMediaType === "image") {
    params.image_url = finalMediaUrl;
    params.media_type = "IMAGE";
  }

  if (finalMediaType === "video") {
    params.video_url = finalMediaUrl;
    params.media_type = "VIDEO";
  }

  const containerResponse = await axios.post(createContainerUrl, null, {
    params,
    timeout: 60000
  });

  const creationId = containerResponse.data?.id;

  if (!creationId) {
    throw new Error("Meta did not return creation_id.");
  }

  if (finalMediaType === "video") {
    await pollMediaContainerStatus(creationId);
  }

  const publishUrl = `${GRAPH_HOST}/${GRAPH_VERSION}/${IG_USER_ID}/media_publish`;

  const publishResponse = await axios.post(publishUrl, null, {
    params: {
      creation_id: creationId,
      access_token: META_ACCESS_TOKEN
    },
    timeout: 60000
  });

  return {
    creationId,
    publishId: publishResponse.data?.id,
    mediaType: finalMediaType,
    mediaUrl: finalMediaUrl
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
You are a social media marketing assistant for artificial trees, artificial flowers, luxury interior decor, and premium visual marketing.

Analyze the provided product media and generate Instagram content.

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
  model = GEMINI_IMAGE_MODEL
}) {
  requireGeminiConfig();

  if (!originalImageUrl || !isPublicHttpsUrl(originalImageUrl)) {
    throw new Error("originalImageUrl must be a public HTTPS URL.");
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

  const tempImagePath = saveBase64TempImage({
    base64Data: inlineData.data,
    mimeType: inlineData.mimeType || inlineData.mime_type || "image/png"
  });

  const normalizedPath = path.join(tempDir, `${randomUUID()}.jpg`);

  await sharp(tempImagePath)
    .rotate()
    .jpeg({
      quality: 92,
      mozjpeg: true
    })
    .toFile(normalizedPath);

  const editedImageUrl = await uploadImageToCloudinary(normalizedPath);

  cleanupFile(tempImagePath);
  cleanupFile(normalizedPath);

  return {
    edited_image_url: editedImageUrl,
    editedImageUrl,
    mediaUrl: editedImageUrl,
    imageUrl: editedImageUrl,
    mediaType: "image",
    model
  };
}

/**
 * Root / Health
 */
app.get("/", (req, res) => {
  res.json({
    ok: true,
    app: "AutoFlow Backend",
    status: "running",
    graphHost: GRAPH_HOST,
    graphVersion: GRAPH_VERSION,
    cloudinaryConfigured: Boolean(
      CLOUDINARY_CLOUD_NAME && CLOUDINARY_API_KEY && CLOUDINARY_API_SECRET
    )
  });
});

app.get("/api/health", (req, res) => {
  res.json({
    ok: true,
    timestamp: new Date().toISOString(),
    graphHost: GRAPH_HOST,
    graphVersion: GRAPH_VERSION,
    cloudinaryConfigured: Boolean(
      CLOUDINARY_CLOUD_NAME && CLOUDINARY_API_KEY && CLOUDINARY_API_SECRET
    )
  });
});

/**
 * Meta test
 */
async function testMetaConnection() {
  requireMetaConfig();

  const url = `${GRAPH_HOST}/${GRAPH_VERSION}/me`;

  const response = await axios.get(url, {
    params: {
      fields: "user_id,username",
      access_token: META_ACCESS_TOKEN
    },
    timeout: 30000
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
    res.json({ ok: true, ...result });
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
    res.json({ ok: true, ...result });
  } catch (error) {
    console.error("Meta test failed:", error.response?.data || error.message);
    res.status(500).json({
      ok: false,
      error: error.response?.data || error.message
    });
  }
});

/**
 * Upload media to Cloudinary
 */
app.post("/api/upload", upload.any(), async (req, res) => {
  try {
    requireCloudinaryConfig();

    const file = req.files?.[0];

    if (!file) {
      return res.status(400).json({
        ok: false,
        error: "No media file uploaded. Send multipart/form-data with any file field."
      });
    }

    const mediaType = detectMediaTypeFromFile(file);

    if (!mediaType) {
      cleanupFile(file.path);
      return res.status(400).json({
        ok: false,
        error: "Unsupported file type. Upload image or video only."
      });
    }

    let uploaded;

    if (mediaType === "image") {
      uploaded = await uploadNormalizedImage(file);
    } else {
      uploaded = await uploadVideo(file);
    }

    res.json({
      ok: true,
      url: uploaded.mediaUrl,
      mediaUrl: uploaded.mediaUrl,
      media_url: uploaded.mediaUrl,
      imageUrl: uploaded.imageUrl,
      image_url: uploaded.imageUrl,
      videoUrl: uploaded.videoUrl,
      video_url: uploaded.videoUrl,
      mediaType: uploaded.mediaType,
      media_type: uploaded.mediaType,
      mimeType: uploaded.mimeType,
      mime_type: uploaded.mimeType
    });
  } catch (error) {
    console.error("Upload failed:", error.response?.data || error.message);

    if (req.files) {
      for (const file of req.files) {
        cleanupFile(file.path);
      }
    }

    res.status(500).json({
      ok: false,
      error: error.response?.data || error.message
    });
  }
});

/**
 * Debug inspect URL
 */
app.post("/api/debug/inspect-url", async (req, res) => {
  try {
    const url = req.body.url;

    if (!url || !isPublicHttpsUrl(url)) {
      return res.status(400).json({
        ok: false,
        error: "url must be a public HTTPS URL."
      });
    }

    let response;

    try {
      response = await axios.head(url, {
        maxRedirects: 5,
        timeout: 30000,
        validateStatus: () => true
      });
    } catch {
      response = await axios.get(url, {
        maxRedirects: 5,
        timeout: 30000,
        responseType: "stream",
        validateStatus: () => true
      });
    }

    res.json({
      ok: true,
      status: response.status,
      contentType: response.headers["content-type"] || null,
      contentLength: response.headers["content-length"] || null,
      finalUrl: response.request?.res?.responseUrl || url
    });
  } catch (error) {
    res.status(500).json({
      ok: false,
      error: error.response?.data || error.message
    });
  }
});

/**
 * Publish Now
 */
app.post("/api/meta/publish-now", async (req, res) => {
  try {
    const { mediaUrl, imageUrl, videoUrl, mediaType } = normalizeMediaInput(req.body);
    const caption = req.body.caption || req.body.final_text || req.body.finalText || "";

    const result = await publishToInstagram({
      mediaUrl,
      imageUrl,
      videoUrl,
      mediaType,
      caption
    });

    res.json({
      ok: true,
      result
    });
  } catch (error) {
    console.error("Publish now failed:", error.response?.data || error.message);

    const metaError = error.response?.data || error.message;

    res.status(500).json({
      ok: false,
      error: metaError,
      friendlyMessage:
        error.response?.data?.error?.code === 9004
          ? "Meta could not fetch this media URL. Re-upload the file and make sure Cloudinary URL is used."
          : undefined
    });
  }
});

/**
 * Gemini models
 */
app.post("/api/gemini/list-models", async (req, res) => {
  try {
    const apiKey = req.body.apiKey || GEMINI_API_KEY;

    if (!apiKey) {
      return res.status(400).json({
        ok: false,
        error: "Gemini API key is required."
      });
    }

    const response = await axios.get(
      "https://generativelanguage.googleapis.com/v1beta/models",
      {
        params: { key: apiKey },
        timeout: 30000
      }
    );

    const models = response.data.models || [];

    const normalizedModels = models.map((model) => {
      const fullName = model.name || "";
      const shortName = fullName.replace("models/", "");

      return {
        name: fullName,
        id: shortName,
        displayName: model.displayName || shortName,
        description: model.description || "",
        supportedGenerationMethods: model.supportedGenerationMethods || []
      };
    });

    const textModels = normalizedModels
      .filter((model) => {
        const methods = model.supportedGenerationMethods || [];
        const id = model.id.toLowerCase();

        return (
          methods.includes("generateContent") &&
          !id.includes("embedding") &&
          !id.includes("aqa")
        );
      })
      .map((model) => model.id);

    const imageModels = normalizedModels
      .filter((model) => {
        const methods = model.supportedGenerationMethods || [];
        const id = model.id.toLowerCase();

        return (
          methods.includes("generateContent") &&
          (id.includes("image") ||
            id.includes("imagen") ||
            id.includes("flash-image") ||
            id.includes("nano"))
        );
      })
      .map((model) => model.id);

    res.json({
      ok: true,
      models: normalizedModels,
      textModels,
      imageModels
    });
  } catch (error) {
    console.error("Gemini list models failed:", error.response?.data || error.message);

    res.status(500).json({
      ok: false,
      error: error.response?.data || error.message
    });
  }
});

app.post("/api/gemini/test", async (req, res) => {
  try {
    requireGeminiConfig();

    const model = req.body.model || GEMINI_TEXT_MODEL;

    const text = await generateTextWithGemini(
      'Return strict JSON only: {"ok": true, "message": "Gemini connected"}',
      model
    );

    res.json({
      ok: true,
      raw: text,
      result: parseJsonLoose(text) || { message: text },
      model
    });
  } catch (error) {
    console.error("Gemini test failed:", error.response?.data || error.message);
    res.status(500).json({
      ok: false,
      error: error.response?.data || error.message
    });
  }
});

/**
 * Caption
 */
app.post("/api/gemini/generate-caption", async (req, res) => {
  try {
    const imageUrl =
      getBodyValue(req.body, "imageUrl", "image_url") ||
      getBodyValue(req.body, "mediaUrl", "media_url");

    const language = req.body.language || "arabic";
    const tone = req.body.tone || "premium";
    const model = req.body.model || req.body.selectedModel || GEMINI_TEXT_MODEL;

    if (!imageUrl) {
      return res.status(400).json({
        ok: false,
        error: "imageUrl or mediaUrl is required."
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
    console.error("Generate caption failed:", error.response?.data || error.message);
    res.status(500).json({
      ok: false,
      error: error.response?.data || error.message
    });
  }
});

/**
 * Generate edit prompt
 */
app.post("/api/gemini/generate-edit-prompt", async (req, res) => {
  try {
    const imageUrl =
      getBodyValue(req.body, "imageUrl", "image_url") ||
      getBodyValue(req.body, "mediaUrl", "media_url");

    const editStyle =
      req.body.editStyle || req.body.edit_style || "luxury interior background";

    const language = req.body.language || "english";
    const model = req.body.model || req.body.selectedModel || GEMINI_TEXT_MODEL;

    if (!imageUrl) {
      return res.status(400).json({
        ok: false,
        error: "imageUrl or mediaUrl is required."
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
    console.error("Generate edit prompt failed:", error.response?.data || error.message);
    res.status(500).json({
      ok: false,
      error: error.response?.data || error.message
    });
  }
});

/**
 * AI image editing
 */
app.post("/api/ai/edit-image", async (req, res) => {
  try {
    const originalImageUrl =
      getBodyValue(req.body, "originalImageUrl", "original_image_url") ||
      getBodyValue(req.body, "imageUrl", "image_url") ||
      getBodyValue(req.body, "mediaUrl", "media_url");

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
      model
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
 * Create scheduled post
 */
app.post("/api/posts", (req, res) => {
  const { mediaUrl, imageUrl, videoUrl, mediaType } = normalizeMediaInput(req.body);

  const caption = req.body.caption || "";
  const hashtags = normalizeHashtags(req.body.hashtags);
  const scheduledAt = getBodyValue(req.body, "scheduledAt", "scheduled_at");

  if (!mediaUrl || !scheduledAt) {
    return res.status(400).json({
      ok: false,
      error: "mediaUrl and scheduledAt are required."
    });
  }

  if (mediaType !== "image" && mediaType !== "video") {
    return res.status(400).json({
      ok: false,
      error: "mediaType must be image or video."
    });
  }

  const now = new Date().toISOString();
  const finalText = `${caption}\n${hashtags.join(" ")}`.trim();

  const post = {
    id: randomUUID(),

    mediaUrl,
    media_url: mediaUrl,

    imageUrl,
    image_url: imageUrl,

    videoUrl,
    video_url: videoUrl,

    mediaType,
    media_type: mediaType,

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

app.post("/api/posts/:id/retry", async (req, res) => {
  const post = posts.find((item) => item.id === req.params.id);

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
      mediaUrl: post.mediaUrl,
      imageUrl: post.imageUrl,
      videoUrl: post.videoUrl,
      mediaType: post.mediaType,
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
 * Cron scheduler
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
        mediaUrl: post.mediaUrl,
        imageUrl: post.imageUrl,
        videoUrl: post.videoUrl,
        mediaType: post.mediaType,
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
