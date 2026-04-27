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
import { createClient } from "@supabase/supabase-js";

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
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;
const AI_DEFAULT_PROVIDER = process.env.AI_DEFAULT_PROVIDER || "gemini";

const GEMINI_TEXT_MODEL = process.env.GEMINI_TEXT_MODEL || "gemini-2.5-flash";
const GEMINI_IMAGE_MODEL =
  process.env.GEMINI_IMAGE_MODEL || "gemini-2.5-flash-image-preview";

const DEFAULT_BUSINESS_NAME = process.env.DEFAULT_BUSINESS_NAME || "Flower Center";
const DEFAULT_LOCATION = process.env.DEFAULT_LOCATION || "UAE";
const DEFAULT_CTA = process.env.DEFAULT_CTA || "Contact us today";
const DEFAULT_HASHTAG_COUNT = Number(process.env.DEFAULT_HASHTAG_COUNT || 10);

const CLOUDINARY_CLOUD_NAME = process.env.CLOUDINARY_CLOUD_NAME;
const CLOUDINARY_API_KEY = process.env.CLOUDINARY_API_KEY;
const CLOUDINARY_API_SECRET = process.env.CLOUDINARY_API_SECRET;

const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;

cloudinary.config({
  cloud_name: CLOUDINARY_CLOUD_NAME,
  api_key: CLOUDINARY_API_KEY,
  api_secret: CLOUDINARY_API_SECRET,
  secure: true
});

const supabase =
  SUPABASE_URL && SUPABASE_SERVICE_ROLE_KEY
    ? createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    : null;

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

app.use((req, res, next) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.originalUrl}`);
  next();
});

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function requireMetaConfig() {
  if (!META_ACCESS_TOKEN || !IG_USER_ID) {
    throw new Error("Missing META_ACCESS_TOKEN or IG_USER_ID.");
  }
}

function requireGeminiConfig(apiKey = GEMINI_API_KEY) {
  if (!apiKey) {
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

function requireSupabaseConfig() {
  if (!supabase) {
    throw new Error(
      "Missing Supabase environment variables: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY."
    );
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

function isPublicHttpsUrl(url) {
  try {
    const parsed = new URL(url);
    return parsed.protocol === "https:";
  } catch {
    return false;
  }
}

function isValidUuid(value) {
  if (!value) return false;

  return /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(
    String(value).trim()
  );
}

function firstDefined(...values) {
  for (const value of values) {
    if (value !== undefined && value !== null && String(value).trim() !== "") {
      return value;
    }
  }

  return null;
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
  const clean = String(url || "").split("?")[0].toLowerCase();

  if (
    clean.endsWith(".jpg") ||
    clean.endsWith(".jpeg") ||
    clean.endsWith(".png") ||
    clean.endsWith(".webp")
  ) {
    return "image";
  }

  if (
    clean.endsWith(".mp4") ||
    clean.endsWith(".mov") ||
    clean.endsWith(".m4v")
  ) {
    return "video";
  }

  return null;
}

function detectMediaTypeFromFile(file) {
  const mime = String(file.mimetype || "").toLowerCase();
  const name = String(file.originalname || "").toLowerCase();

  if (mime.startsWith("image/")) return "image";
  if (mime.startsWith("video/")) return "video";

  if (
    name.endsWith(".jpg") ||
    name.endsWith(".jpeg") ||
    name.endsWith(".png") ||
    name.endsWith(".webp") ||
    name.endsWith(".heic")
  ) {
    return "image";
  }

  if (
    name.endsWith(".mp4") ||
    name.endsWith(".mov") ||
    name.endsWith(".m4v")
  ) {
    return "video";
  }

  return null;
}

function normalizeProvider(value) {
  const provider = String(value || AI_DEFAULT_PROVIDER || "gemini")
    .trim()
    .toLowerCase();

  if (["gemini", "openai", "openrouter"].includes(provider)) {
    return provider;
  }

  return "gemini";
}

function getProviderApiKey({ provider, apiKey }) {
  const cleanKey = apiKey ? String(apiKey).trim() : "";

  if (cleanKey.length > 0) return cleanKey;

  if (provider === "gemini") return GEMINI_API_KEY;
  if (provider === "openai") return OPENAI_API_KEY;
  if (provider === "openrouter") return OPENROUTER_API_KEY;

  return null;
}

function classifyModel(provider, modelId, raw = {}) {
  const id = String(modelId || "").replace("models/", "");
  const lower = id.toLowerCase();
  const methods = raw.supportedGenerationMethods || [];

  const isEmbedding =
    lower.includes("embedding") ||
    lower.includes("embed") ||
    lower.includes("aqa");

  const supportsText =
    provider === "gemini"
      ? methods.includes("generateContent") && !isEmbedding
      : !isEmbedding;

  const supportsImage =
    lower.includes("image") ||
    lower.includes("imagen") ||
    lower.includes("vision") ||
    lower.includes("gpt-4o") ||
    lower.includes("nano");

  const supportsVideo = lower.includes("video") || lower.includes("veo");

  let type = "text";

  if (supportsVideo) type = "video";
  else if (supportsImage) type = "image";

  let isFree = false;

  if (provider === "gemini") {
    isFree = true;
  }

  if (provider === "openrouter") {
    const pricing = raw.pricing || {};
    const promptPrice = Number(pricing.prompt || 0);
    const completionPrice = Number(pricing.completion || 0);

    isFree =
      lower.includes(":free") ||
      (promptPrice === 0 && completionPrice === 0);
  }

  return {
    id,
    name: raw.name || id,
    displayName: raw.displayName || raw.name || id,
    description: raw.description || "",
    type,
    supportsText,
    supportsImage,
    supportsVideo,
    isFree,
    supportedGenerationMethods: methods
  };
}

function extractBodySource(body) {
  return body.post || body.item || body.data || body.payload || body;
}

function normalizeMediaInput(body) {
  const source = extractBodySource(body);
  const media = source.media || source.mediaAsset || source.asset || {};

  const mediaAssetId = firstDefined(
    source.mediaAssetId,
    source.media_asset_id,
    source.mediaId,
    source.media_id,
    media.mediaAssetId,
    media.media_asset_id,
    media.id
  );

  const imageUrl = firstDefined(
    source.imageUrl,
    source.image_url,
    media.imageUrl,
    media.image_url
  );

  const videoUrl = firstDefined(
    source.videoUrl,
    source.video_url,
    media.videoUrl,
    media.video_url
  );

  const mediaUrl = firstDefined(
    source.mediaUrl,
    source.media_url,
    source.url,
    source.publicUrl,
    source.public_url,
    media.mediaUrl,
    media.media_url,
    media.url,
    imageUrl,
    videoUrl
  );

  let mediaType = firstDefined(
    source.mediaType,
    source.media_type,
    media.mediaType,
    media.media_type
  );

  mediaType = mediaType ? String(mediaType).toLowerCase() : null;

  if (!mediaType && imageUrl) mediaType = "image";
  if (!mediaType && videoUrl) mediaType = "video";
  if (!mediaType && mediaUrl) mediaType = detectMediaTypeFromUrl(mediaUrl);

  if (mediaType === "photo") mediaType = "image";
  if (mediaType === "reel") mediaType = "video";

  return {
    mediaAssetId,
    mediaUrl,
    imageUrl,
    videoUrl,
    mediaType
  };
}

function normalizeScheduledAt(body) {
  const source = extractBodySource(body);

  return firstDefined(
    source.scheduledAt,
    source.scheduled_at,
    source.scheduleAt,
    source.schedule_at,
    source.scheduledFor,
    source.scheduled_for,
    source.publishAt,
    source.publish_at,
    source.dateTime,
    source.datetime,
    source.date_time,
    source.time
  );
}

function normalizeCaption(body) {
  const source = extractBodySource(body);

  return String(
    firstDefined(source.caption, source.finalText, source.final_text, "") || ""
  );
}

function normalizeInputHashtags(body) {
  const source = extractBodySource(body);
  return normalizeHashtags(source.hashtags || source.tags || []);
}

async function fetchMediaAssetById(mediaAssetId) {
  if (!isValidUuid(mediaAssetId)) {
    console.warn("Skipping invalid mediaAssetId:", mediaAssetId);
    return null;
  }

  requireSupabaseConfig();

  const { data, error } = await supabase
    .from("media_assets")
    .select("*")
    .eq("id", mediaAssetId)
    .maybeSingle();

  if (error) {
    console.warn("Could not fetch media asset:", error.message);
    return null;
  }

  return data;
}

async function resolveMediaAsset({ mediaAssetId, mediaUrl }) {
  requireSupabaseConfig();

  if (isValidUuid(mediaAssetId)) {
    const { data, error } = await supabase
      .from("media_assets")
      .select("*")
      .eq("id", mediaAssetId)
      .maybeSingle();

    if (error) {
      console.warn("Could not fetch media asset by UUID:", error.message);
    }

    if (data) return data;
  } else if (mediaAssetId) {
    console.warn("Ignoring non-UUID mediaAssetId:", mediaAssetId);
  }

  if (mediaUrl) {
    const { data, error } = await supabase
      .from("media_assets")
      .select("*")
      .eq("media_url", mediaUrl)
      .maybeSingle();

    if (error) {
      console.warn("Could not fetch media asset by media_url:", error.message);
    }

    if (data) return data;
  }

  return null;
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
  console.log(`Waiting for Meta container to be ready: ${containerId}`);

  await sleep(3000);

  for (let attempt = 1; attempt <= 24; attempt += 1) {
    try {
      const response = await axios.get(
        `${GRAPH_HOST}/${GRAPH_VERSION}/${containerId}`,
        {
          params: {
            fields: "status_code,status",
            access_token: META_ACCESS_TOKEN
          },
          timeout: 30000
        }
      );

      const statusCode = response.data?.status_code;
      const status = response.data?.status;

      console.log(
        `Meta container ${containerId} status attempt ${attempt}:`,
        statusCode || status || response.data
      );

      if (statusCode === "FINISHED") {
        return response.data;
      }

      if (statusCode === "ERROR") {
        throw new Error(
          `Meta media container failed: ${status || "Unknown error"}`
        );
      }

      await sleep(5000);
    } catch (error) {
      console.warn(
        `Meta container status check failed attempt ${attempt}:`,
        error.response?.data || error.message
      );

      if (attempt === 24) throw error;

      await sleep(5000);
    }
  }

  throw new Error("Meta media container did not finish processing in time.");
}

async function publishToInstagram({
  mediaUrl,
  imageUrl,
  videoUrl,
  mediaType,
  caption
}) {
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

  console.log("Creating Meta media container:", {
    mediaType: finalMediaType,
    mediaUrl: finalMediaUrl
  });

  const containerResponse = await axios.post(createContainerUrl, null, {
    params,
    timeout: 60000
  });

  const creationId = containerResponse.data?.id;

  if (!creationId) {
    throw new Error("Meta did not return creation_id.");
  }

  console.log(`Meta media container created: ${creationId}`);

  await pollMediaContainerStatus(creationId);

  const publishUrl = `${GRAPH_HOST}/${GRAPH_VERSION}/${IG_USER_ID}/media_publish`;

  let publishResponse = null;

  for (let attempt = 1; attempt <= 3; attempt += 1) {
    try {
      console.log(`Publishing Meta container ${creationId}, attempt ${attempt}`);

      publishResponse = await axios.post(publishUrl, null, {
        params: {
          creation_id: creationId,
          access_token: META_ACCESS_TOKEN
        },
        timeout: 60000
      });

      break;
    } catch (error) {
      const metaCode = error.response?.data?.error?.code;

      console.warn(
        `Meta publish failed attempt ${attempt}:`,
        error.response?.data || error.message
      );

      if (metaCode === 9007 && attempt < 3) {
        console.warn("Meta media not ready. Retrying publish in 10 seconds...");
        await sleep(10000);
        await pollMediaContainerStatus(creationId);
        continue;
      }

      throw error;
    }
  }

  if (!publishResponse?.data?.id) {
    throw new Error("Meta did not return publish ID.");
  }

  return {
    creationId,
    publishId: publishResponse.data.id,
    mediaType: finalMediaType,
    mediaUrl: finalMediaUrl
  };
}

function getPresetInstruction(captionPreset, customPrompt) {
  const preset = String(captionPreset || "Luxury Product Caption").trim();

  if (preset === "Custom Prompt") {
    return customPrompt && String(customPrompt).trim()
      ? String(customPrompt).trim()
      : "Write a premium Instagram caption based on the visible product and scene.";
  }

  const presets = {
    "Luxury Product Caption":
      "Focus on premium decor, elegance, high-end styling, luxury ambience, and refined taste.",
    "Artificial Tree Marketing":
      "Focus on realistic artificial trees, no maintenance, custom sizes, greenery styling, and indoor/outdoor decor.",
    "Artificial Flower Arrangement":
      "Focus on flower colors, arrangement style, luxury floral styling, events, interiors, and decorative impact.",
    "Interior Design Decor":
      "Focus on how the product improves the interior space, ambience, warmth, balance, and luxury decor.",
    "Villa / Entrance Decor":
      "Focus on villa entrances, majlis, welcoming first impression, elegant greenery, and premium home decor.",
    "Hotel / Mall / Commercial Decor":
      "Focus on commercial spaces, hotels, malls, offices, durability, visual impact, and professional installation.",
    "Short Premium Caption":
      "Write one short, elegant, direct caption. Keep it premium and minimal.",
    "Arabic Social Media Caption":
      "Write natural Arabic social media copy. Make it premium but not stiff or over-formal.",
    "Before / After Style":
      "Write like a transformation post. Emphasize how the decor changes the feeling of the space."
  };

  return presets[preset] || presets["Luxury Product Caption"];
}

function buildCaptionPrompt({
  language,
  tone,
  captionPreset,
  customPrompt,
  businessName,
  location,
  cta,
  hashtagCount
}) {
  const presetInstruction = getPresetInstruction(captionPreset, customPrompt);
  const safeHashtagCount = Number.isFinite(Number(hashtagCount))
    ? Math.max(3, Math.min(25, Number(hashtagCount)))
    : DEFAULT_HASHTAG_COUNT;

  return `
You are a senior Instagram marketing copywriter for ${businessName}, a ${location}-based company specializing in premium artificial trees, artificial flowers, custom greenery, and luxury decor installations.

Analyze the image carefully before writing.

Identify what is visible:
- Is it an artificial tree, flowers, greenery wall, planter, staircase decor, entrance decor, villa decor, commercial decor, or another product?
- What colors are visible?
- What style does the space or product suggest?
- Is it luxury, modern, natural, minimal, classic, or commercial?
- What customer would want this?
- What is the strongest marketing angle?

Caption preset:
${captionPreset}

Preset instruction:
${presetInstruction}

Language:
${language}

Tone:
${tone}

Business name:
${businessName}

Location:
${location}

CTA:
${cta}

Hashtag count:
${safeHashtagCount}

Write a caption that matches the actual visible image, not a generic caption.

Return strict JSON only:
{
  "caption": "caption here",
  "hashtags": ["#hashtag1", "#hashtag2"],
  "alt_text": "short alt text",
  "detected_product": "what the image shows",
  "visual_description": "short visual analysis",
  "marketing_angle": "main sales angle"
}

Rules:
- Caption must be specific to the image.
- Do not use placeholders.
- Do not write "#hashtag1".
- Do not invent discounts, offers, guarantees, or prices.
- Do not mention AI.
- Do not say "in the image" unless it is necessary.
- Mention artificial tree / artificial flowers only when visually relevant.
- If the product looks like an artificial olive tree, mention artificial olive tree.
- If the product looks like flowers, mention floral arrangement, colors, and styling.
- If the setting looks like an entrance, staircase, villa, hotel, mall, showroom, or commercial decor, mention that appropriately.
- Hashtags must be relevant to UAE decor, artificial trees, artificial flowers, interiors, villas, hotels, landscaping, or the detected product.
- Use no more than ${safeHashtagCount} hashtags.
- Arabic captions must sound natural, premium, and social-media-ready, not literal translation.
- English captions must be clean, premium, and concise.
- If language is Arabic + English, write a short Arabic caption followed by a short English line.
`;
}

async function generateTextWithGemini(
  prompt,
  modelName = GEMINI_TEXT_MODEL,
  apiKey = GEMINI_API_KEY
) {
  requireGeminiConfig(apiKey);

  const genAI = new GoogleGenerativeAI(apiKey);
  const model = genAI.getGenerativeModel({ model: modelName });

  const result = await model.generateContent(prompt);
  return result.response.text();
}

async function generateCaptionWithGeminiVision({
  imageUrl,
  prompt,
  modelName = GEMINI_TEXT_MODEL,
  apiKey = GEMINI_API_KEY,
  fallbackTextOnly = false
}) {
  requireGeminiConfig(apiKey);

  const genAI = new GoogleGenerativeAI(apiKey);
  const model = genAI.getGenerativeModel({ model: modelName });

  try {
    const imagePart = await fetchImageAsInlineData(imageUrl);
    const result = await model.generateContent([prompt, imagePart]);
    return result.response.text();
  } catch (error) {
    console.error("Gemini image analysis failed:", error.response?.data || error.message);

    if (!fallbackTextOnly) {
      throw new Error("Could not fetch image for visual analysis.");
    }

    const result = await model.generateContent(`${prompt}\n\nImage URL: ${imageUrl}`);
    return result.response.text();
  }
}

async function generateCaptionWithOpenAI({
  provider,
  apiKey,
  model,
  imageUrl,
  prompt
}) {
  const endpoint =
    provider === "openrouter"
      ? "https://openrouter.ai/api/v1/chat/completions"
      : "https://api.openai.com/v1/chat/completions";

  const response = await axios.post(
    endpoint,
    {
      model,
      messages: [
        {
          role: "user",
          content: [
            {
              type: "text",
              text: prompt
            },
            {
              type: "image_url",
              image_url: {
                url: imageUrl
              }
            }
          ]
        }
      ]
    },
    {
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json"
      },
      timeout: 90000
    }
  );

  return response.data?.choices?.[0]?.message?.content || "";
}

async function generateCaptionWithAI({
  provider = "gemini",
  apiKey,
  imageUrl,
  language = "arabic",
  tone = "premium",
  model,
  captionPreset = "Luxury Product Caption",
  customPrompt = "",
  businessName = DEFAULT_BUSINESS_NAME,
  location = DEFAULT_LOCATION,
  cta = DEFAULT_CTA,
  hashtagCount = DEFAULT_HASHTAG_COUNT,
  fallbackTextOnly = false
}) {
  const selectedProvider = normalizeProvider(provider);
  const selectedApiKey = getProviderApiKey({
    provider: selectedProvider,
    apiKey
  });

  if (!selectedApiKey) {
    throw new Error(`Missing API key for provider: ${selectedProvider}`);
  }

  const prompt = buildCaptionPrompt({
    language,
    tone,
    captionPreset,
    customPrompt,
    businessName,
    location,
    cta,
    hashtagCount
  });

  let text = "";

  if (selectedProvider === "gemini") {
    text = await generateCaptionWithGeminiVision({
      imageUrl,
      prompt,
      modelName: model || GEMINI_TEXT_MODEL,
      apiKey: selectedApiKey,
      fallbackTextOnly
    });
  } else {
    text = await generateCaptionWithOpenAI({
      provider: selectedProvider,
      apiKey: selectedApiKey,
      model,
      imageUrl,
      prompt
    });
  }

  const parsed = parseJsonLoose(text);

  if (parsed) {
    return {
      caption: parsed.caption || "",
      hashtags: normalizeHashtags(parsed.hashtags),
      alt_text: parsed.alt_text || "",
      detected_product: parsed.detected_product || "",
      visual_description: parsed.visual_description || "",
      marketing_angle: parsed.marketing_angle || ""
    };
  }

  return {
    caption: text,
    hashtags: [],
    alt_text: "",
    detected_product: "",
    visual_description: "",
    marketing_angle: ""
  };
}

async function generateEditPromptWithGemini({
  imageUrl,
  editStyle = "luxury interior background",
  language = "english",
  model = GEMINI_TEXT_MODEL,
  apiKey = GEMINI_API_KEY
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

  const text = await generateCaptionWithGeminiVision({
    imageUrl,
    prompt,
    modelName: model,
    apiKey,
    fallbackTextOnly: false
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
  requireGeminiConfig(GEMINI_API_KEY);

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
        parts: [{ text: prompt }, imagePart]
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

async function markMediaPublished(mediaAssetId, publishedAt) {
  if (!mediaAssetId || !isValidUuid(mediaAssetId)) return;

  requireSupabaseConfig();

  await supabase
    .from("media_assets")
    .update({
      is_published: true,
      published_at: publishedAt,
      updated_at: new Date().toISOString()
    })
    .eq("id", mediaAssetId);
}

async function publishScheduledPost(post) {
  requireSupabaseConfig();

  const nowIso = new Date().toISOString();

  await supabase
    .from("scheduled_posts")
    .update({
      status: "publishing",
      publish_attempts: (post.publish_attempts || 0) + 1,
      updated_at: nowIso
    })
    .eq("id", post.id);

  try {
    const result = await publishToInstagram({
      mediaUrl: post.media_url,
      imageUrl: post.image_url,
      videoUrl: post.video_url,
      mediaType: post.media_type,
      caption: post.final_text || post.caption || ""
    });

    const publishedAt = new Date().toISOString();

    const { data: updatedPost, error: updateError } = await supabase
      .from("scheduled_posts")
      .update({
        status: "published",
        meta_container_id: result.creationId,
        meta_publish_id: result.publishId,
        published_at: publishedAt,
        error_message: null,
        updated_at: publishedAt
      })
      .eq("id", post.id)
      .select()
      .single();

    if (updateError) throw updateError;

    await markMediaPublished(post.media_asset_id, publishedAt);

    return updatedPost;
  } catch (error) {
    const errorMessage = JSON.stringify(error.response?.data || error.message);

    await supabase
      .from("scheduled_posts")
      .update({
        status: "failed",
        error_message: errorMessage,
        updated_at: new Date().toISOString()
      })
      .eq("id", post.id);

    throw error;
  }
}

async function createScheduledPostFromInput(inputBody) {
  requireSupabaseConfig();

  let { mediaAssetId, mediaUrl, imageUrl, videoUrl, mediaType } =
    normalizeMediaInput(inputBody);

  const mediaAsset = await resolveMediaAsset({
    mediaAssetId,
    mediaUrl
  });

  if (mediaAsset) {
    mediaAssetId = mediaAsset.id;
    mediaUrl = mediaUrl || mediaAsset.media_url;
    imageUrl = imageUrl || mediaAsset.image_url;
    videoUrl = videoUrl || mediaAsset.video_url;
    mediaType = mediaType || mediaAsset.media_type;
  } else {
    if (!isValidUuid(mediaAssetId)) {
      console.warn("Ignoring non-UUID mediaAssetId:", mediaAssetId);
      mediaAssetId = null;
    }
  }

  const caption = normalizeCaption(inputBody);
  const hashtags = normalizeInputHashtags(inputBody);
  const scheduledAt = normalizeScheduledAt(inputBody);

  if (!mediaUrl || !scheduledAt) {
    const error = new Error("mediaUrl and scheduledAt are required.");
    error.statusCode = 400;
    error.details = {
      received: inputBody,
      parsed: {
        mediaAssetId,
        mediaUrl,
        imageUrl,
        videoUrl,
        mediaType,
        scheduledAt
      }
    };
    throw error;
  }

  if (mediaType !== "image" && mediaType !== "video") {
    const error = new Error("mediaType must be image or video.");
    error.statusCode = 400;
    error.details = {
      received: inputBody,
      parsed: {
        mediaAssetId,
        mediaUrl,
        imageUrl,
        videoUrl,
        mediaType,
        scheduledAt
      }
    };
    throw error;
  }

  const parsedDate = new Date(scheduledAt);

  if (Number.isNaN(parsedDate.getTime())) {
    const error = new Error("scheduledAt is not a valid date.");
    error.statusCode = 400;
    error.details = {
      received: inputBody,
      scheduledAt
    };
    throw error;
  }

  const finalText = `${caption}\n${hashtags.join(" ")}`.trim();

  const insertPayload = {
    media_asset_id: mediaAssetId,
    media_url: mediaUrl,
    image_url: imageUrl,
    video_url: videoUrl,
    media_type: mediaType,
    caption,
    hashtags,
    final_text: finalText,
    scheduled_at: parsedDate.toISOString(),
    status: "approved"
  };

  console.log("Creating scheduled post payload:", insertPayload);

  const { data: post, error } = await supabase
    .from("scheduled_posts")
    .insert(insertPayload)
    .select()
    .single();

  if (error) {
    console.error("Supabase insert scheduled_posts failed:", error);
    throw error;
  }

  if (mediaAssetId && isValidUuid(mediaAssetId)) {
    const { error: mediaUpdateError } = await supabase
      .from("media_assets")
      .update({
        is_scheduled: true,
        scheduled_post_id: post.id,
        updated_at: new Date().toISOString()
      })
      .eq("id", mediaAssetId);

    if (mediaUpdateError) {
      console.warn("Media asset schedule flag update failed:", mediaUpdateError);
    }
  }

  console.log("Scheduled post created:", post.id);

  return post;
}

/**
 * ROOT / HEALTH
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
    ),
    supabaseConfigured: Boolean(supabase)
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
    ),
    supabaseConfigured: Boolean(supabase)
  });
});

/**
 * META TEST
 */
async function testMetaConnection() {
  requireMetaConfig();

  const response = await axios.get(`${GRAPH_HOST}/${GRAPH_VERSION}/me`, {
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
    res.status(500).json({
      ok: false,
      error: error.response?.data || error.message
    });
  }
});

/**
 * MEDIA UPLOAD / LIBRARY
 */
app.post("/api/upload", upload.any(), async (req, res) => {
  try {
    requireCloudinaryConfig();
    requireSupabaseConfig();

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

    const { data: media, error } = await supabase
      .from("media_assets")
      .insert({
        media_url: uploaded.mediaUrl,
        image_url: uploaded.imageUrl,
        video_url: uploaded.videoUrl,
        media_type: uploaded.mediaType,
        mime_type: uploaded.mimeType,
        file_name: file.originalname || null,
        is_uploaded: true,
        is_scheduled: false,
        is_published: false
      })
      .select()
      .single();

    if (error) throw error;

    res.json({
      ok: true,
      media,
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
      for (const file of req.files) cleanupFile(file.path);
    }

    res.status(500).json({
      ok: false,
      error: error.response?.data || error.message
    });
  }
});

app.get("/api/media", async (req, res) => {
  try {
    requireSupabaseConfig();

    const { data, error } = await supabase
      .from("media_assets")
      .select("*")
      .order("created_at", { ascending: false });

    if (error) throw error;

    res.json({
      ok: true,
      media: data || []
    });
  } catch (error) {
    res.status(500).json({
      ok: false,
      error: error.message
    });
  }
});

app.delete("/api/media/:id", async (req, res) => {
  try {
    requireSupabaseConfig();

    const { error } = await supabase
      .from("media_assets")
      .delete()
      .eq("id", req.params.id);

    if (error) throw error;

    res.json({
      ok: true,
      deleted: true
    });
  } catch (error) {
    res.status(500).json({
      ok: false,
      error: error.message
    });
  }
});

/**
 * DEBUG URL
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
 * PUBLISH NOW
 */
app.post("/api/meta/publish-now", async (req, res) => {
  try {
    const { mediaAssetId, mediaUrl, imageUrl, videoUrl, mediaType } =
      normalizeMediaInput(req.body);

    const caption =
      req.body.caption || req.body.final_text || req.body.finalText || "";

    const result = await publishToInstagram({
      mediaUrl,
      imageUrl,
      videoUrl,
      mediaType,
      caption
    });

    if (mediaAssetId && isValidUuid(mediaAssetId)) {
      await markMediaPublished(mediaAssetId, new Date().toISOString());
    }

    res.json({
      ok: true,
      result
    });
  } catch (error) {
    const metaError = error.response?.data || error.message;

    res.status(500).json({
      ok: false,
      error: metaError,
      friendlyMessage:
        error.response?.data?.error?.code === 9004
          ? "Meta could not fetch this media URL. Re-upload the file and make sure Cloudinary URL is used."
          : error.response?.data?.error?.code === 9007
            ? "Meta media was not ready. Backend waited and retried but Meta still rejected it."
            : undefined
    });
  }
});

/**
 * AI MODELS
 */
async function listModelsHandler(req, res, forcedProvider = null) {
  try {
    const provider = normalizeProvider(forcedProvider || req.body.provider);
    const apiKey = getProviderApiKey({
      provider,
      apiKey: req.body.apiKey
    });

    if (!apiKey) {
      return res.status(400).json({
        ok: false,
        error: `Missing API key for provider: ${provider}`
      });
    }

    let normalizedModels = [];

    if (provider === "gemini") {
      const response = await axios.get(
        "https://generativelanguage.googleapis.com/v1beta/models",
        {
          params: { key: apiKey },
          timeout: 30000
        }
      );

      normalizedModels = (response.data.models || []).map((model) => {
        const id = String(model.name || "").replace("models/", "");
        return classifyModel("gemini", id, model);
      });
    }

    if (provider === "openai") {
      const response = await axios.get("https://api.openai.com/v1/models", {
        headers: { Authorization: `Bearer ${apiKey}` },
        timeout: 30000
      });

      normalizedModels = (response.data.data || []).map((model) =>
        classifyModel("openai", model.id, {
          name: model.id,
          displayName: model.id
        })
      );
    }

    if (provider === "openrouter") {
      const response = await axios.get("https://openrouter.ai/api/v1/models", {
        headers: { Authorization: `Bearer ${apiKey}` },
        timeout: 30000
      });

      normalizedModels = (response.data.data || []).map((model) =>
        classifyModel("openrouter", model.id, {
          name: model.id,
          displayName: model.name || model.id,
          description: model.description || "",
          pricing: model.pricing || {}
        })
      );
    }

    res.json({
      ok: true,
      provider,
      models: normalizedModels,
      textModels: normalizedModels.filter((m) => m.supportsText).map((m) => m.id),
      imageModels: normalizedModels.filter((m) => m.supportsImage).map((m) => m.id),
      videoModels: normalizedModels.filter((m) => m.supportsVideo).map((m) => m.id),
      freeModels: normalizedModels.filter((m) => m.isFree).map((m) => m.id)
    });
  } catch (error) {
    console.error("AI list models failed:", error.response?.data || error.message);

    res.status(500).json({
      ok: false,
      error: error.response?.data || error.message
    });
  }
}

app.post("/api/ai/list-models", (req, res) => listModelsHandler(req, res));
app.post("/api/gemini/list-models", (req, res) =>
  listModelsHandler(req, res, "gemini")
);

app.post("/api/ai/test-model", async (req, res) => {
  try {
    const provider = normalizeProvider(req.body.provider);
    const model = String(req.body.model || "").trim();

    const apiKey = getProviderApiKey({
      provider,
      apiKey: req.body.apiKey
    });

    if (!apiKey) {
      return res.status(400).json({
        ok: false,
        error: `Missing API key for provider: ${provider}`
      });
    }

    if (!model) {
      return res.status(400).json({
        ok: false,
        error: "model is required."
      });
    }

    if (provider === "gemini") {
      const response = await axios.post(
        `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`,
        {
          contents: [
            {
              role: "user",
              parts: [
                {
                  text: 'Return strict JSON only: {"ok": true, "message": "AI connected"}'
                }
              ]
            }
          ]
        },
        {
          params: { key: apiKey },
          headers: { "Content-Type": "application/json" },
          timeout: 60000
        }
      );

      return res.json({
        ok: true,
        provider,
        model,
        raw: response.data
      });
    }

    const endpoint =
      provider === "openrouter"
        ? "https://openrouter.ai/api/v1/chat/completions"
        : "https://api.openai.com/v1/chat/completions";

    const response = await axios.post(
      endpoint,
      {
        model,
        messages: [
          {
            role: "user",
            content:
              'Return strict JSON only: {"ok": true, "message": "AI connected"}'
          }
        ]
      },
      {
        headers: {
          Authorization: `Bearer ${apiKey}`,
          "Content-Type": "application/json"
        },
        timeout: 60000
      }
    );

    res.json({
      ok: true,
      provider,
      model,
      raw: response.data
    });
  } catch (error) {
    res.status(500).json({
      ok: false,
      error: error.response?.data || error.message
    });
  }
});

app.post("/api/gemini/test", async (req, res) => {
  try {
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
    res.status(500).json({
      ok: false,
      error: error.response?.data || error.message
    });
  }
});

/**
 * CAPTION / EDIT PROMPT / IMAGE EDIT
 */
app.post("/api/gemini/generate-caption", async (req, res) => {
  try {
    const imageUrl =
      getBodyValue(req.body, "imageUrl", "image_url") ||
      getBodyValue(req.body, "mediaUrl", "media_url");

    if (!imageUrl) {
      return res.status(400).json({
        ok: false,
        error: "imageUrl or mediaUrl is required."
      });
    }

    const result = await generateCaptionWithAI({
      provider: req.body.provider || "gemini",
      apiKey: req.body.apiKey,
      imageUrl,
      language: req.body.language || "arabic",
      tone: req.body.tone || "premium",
      model: req.body.model || GEMINI_TEXT_MODEL,
      captionPreset: req.body.captionPreset || req.body.caption_preset || "Luxury Product Caption",
      customPrompt: req.body.customPrompt || req.body.custom_prompt || "",
      businessName: req.body.businessName || req.body.business_name || DEFAULT_BUSINESS_NAME,
      location: req.body.location || DEFAULT_LOCATION,
      cta: req.body.cta || DEFAULT_CTA,
      hashtagCount: req.body.hashtagCount || req.body.hashtag_count || DEFAULT_HASHTAG_COUNT,
      fallbackTextOnly: req.body.fallbackTextOnly === true
    });

    const mediaAssetId = req.body.mediaAssetId || req.body.media_asset_id;

    if (isValidUuid(mediaAssetId)) {
      await supabase
        .from("media_assets")
        .update({
          caption: result.caption,
          hashtags: result.hashtags,
          updated_at: new Date().toISOString()
        })
        .eq("id", mediaAssetId);
    }

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

app.post("/api/gemini/generate-edit-prompt", async (req, res) => {
  try {
    const imageUrl =
      getBodyValue(req.body, "imageUrl", "image_url") ||
      getBodyValue(req.body, "mediaUrl", "media_url");

    if (!imageUrl) {
      return res.status(400).json({
        ok: false,
        error: "imageUrl or mediaUrl is required."
      });
    }

    const provider = normalizeProvider(req.body.provider || "gemini");
    const apiKey = getProviderApiKey({
      provider,
      apiKey: req.body.apiKey
    });

    const result = await generateEditPromptWithGemini({
      imageUrl,
      editStyle:
        req.body.editStyle || req.body.edit_style || "luxury interior background",
      language: req.body.language || "english",
      model: req.body.model || GEMINI_TEXT_MODEL,
      apiKey
    });

    res.json({
      ok: true,
      result
    });
  } catch (error) {
    res.status(500).json({
      ok: false,
      error: error.response?.data || error.message
    });
  }
});

app.post("/api/ai/edit-image", async (req, res) => {
  try {
    const originalImageUrl =
      getBodyValue(req.body, "originalImageUrl", "original_image_url") ||
      getBodyValue(req.body, "imageUrl", "image_url") ||
      getBodyValue(req.body, "mediaUrl", "media_url");

    if (!originalImageUrl) {
      return res.status(400).json({
        ok: false,
        error: "originalImageUrl is required."
      });
    }

    const prompt = req.body.prompt || req.body.ai_edit_prompt || "";

    if (!prompt) {
      return res.status(400).json({
        ok: false,
        error: "prompt is required."
      });
    }

    const result = await editImageWithGemini({
      originalImageUrl,
      prompt,
      model: req.body.model || GEMINI_IMAGE_MODEL
    });

    res.json({
      ok: true,
      result
    });
  } catch (error) {
    res.status(500).json({
      ok: false,
      error: error.response?.data || error.message
    });
  }
});

/**
 * POSTS
 */
app.post("/api/posts", async (req, res) => {
  try {
    requireSupabaseConfig();

    console.log("Create scheduled post body:", JSON.stringify(req.body));

    const isBulk =
      Array.isArray(req.body) ||
      Array.isArray(req.body.posts) ||
      Array.isArray(req.body.items);

    if (isBulk) {
      const items = Array.isArray(req.body)
        ? req.body
        : req.body.posts || req.body.items;

      const created = [];
      const failed = [];

      for (const item of items) {
        try {
          const post = await createScheduledPostFromInput(item);
          created.push(post);
        } catch (error) {
          failed.push({
            item,
            error: error.message,
            details: error.details || null
          });
        }
      }

      return res.json({
        ok: failed.length === 0,
        posts: created,
        createdCount: created.length,
        failedCount: failed.length,
        failed
      });
    }

    const post = await createScheduledPostFromInput(req.body);

    res.json({
      ok: true,
      post
    });
  } catch (error) {
    console.error("Create scheduled post failed:", {
      body: req.body,
      error: error.response?.data || error.message || error,
      details: error.details || null
    });

    res.status(error.statusCode || 500).json({
      ok: false,
      error: error.response?.data || error.message || String(error),
      details: error.details || null
    });
  }
});

app.get("/api/posts", async (req, res) => {
  try {
    requireSupabaseConfig();

    const { data, error } = await supabase
      .from("scheduled_posts")
      .select("*")
      .order("scheduled_at", { ascending: true });

    if (error) throw error;

    res.json({
      ok: true,
      posts: data || []
    });
  } catch (error) {
    res.status(500).json({
      ok: false,
      error: error.message
    });
  }
});

app.patch("/api/posts/:id", async (req, res) => {
  try {
    requireSupabaseConfig();

    const update = {};

    if (req.body.caption !== undefined) update.caption = req.body.caption;

    if (req.body.hashtags !== undefined) {
      update.hashtags = normalizeHashtags(req.body.hashtags);
    }

    if (req.body.finalText !== undefined || req.body.final_text !== undefined) {
      update.final_text = req.body.finalText || req.body.final_text;
    }

    if (req.body.scheduledAt !== undefined || req.body.scheduled_at !== undefined) {
      update.scheduled_at = req.body.scheduledAt || req.body.scheduled_at;
    }

    if (req.body.status !== undefined) update.status = req.body.status;

    update.updated_at = new Date().toISOString();

    const { data, error } = await supabase
      .from("scheduled_posts")
      .update(update)
      .eq("id", req.params.id)
      .select()
      .single();

    if (error) throw error;

    res.json({
      ok: true,
      post: data
    });
  } catch (error) {
    res.status(500).json({
      ok: false,
      error: error.message
    });
  }
});

app.delete("/api/posts/:id", async (req, res) => {
  try {
    requireSupabaseConfig();

    const { data, error } = await supabase
      .from("scheduled_posts")
      .update({
        status: "cancelled",
        updated_at: new Date().toISOString()
      })
      .eq("id", req.params.id)
      .select()
      .single();

    if (error) throw error;

    res.json({
      ok: true,
      post: data
    });
  } catch (error) {
    res.status(500).json({
      ok: false,
      error: error.message
    });
  }
});

app.post("/api/posts/:id/retry", async (req, res) => {
  try {
    requireSupabaseConfig();

    const { data: post, error } = await supabase
      .from("scheduled_posts")
      .select("*")
      .eq("id", req.params.id)
      .single();

    if (error) throw error;

    const updatedPost = await publishScheduledPost(post);

    res.json({
      ok: true,
      post: updatedPost
    });
  } catch (error) {
    res.status(500).json({
      ok: false,
      error: error.response?.data || error.message
    });
  }
});

/**
 * CRON SCHEDULER
 */
cron.schedule("*/5 * * * *", async () => {
  try {
    if (!supabase) {
      console.warn("Supabase not configured. Cron skipped.");
      return;
    }

    const nowIso = new Date().toISOString();

    const { data: duePosts, error } = await supabase
      .from("scheduled_posts")
      .select("*")
      .in("status", ["approved", "scheduled"])
      .lte("scheduled_at", nowIso)
      .lt("publish_attempts", 3)
      .order("scheduled_at", { ascending: true })
      .limit(10);

    if (error) throw error;

    if (!duePosts || duePosts.length === 0) {
      return;
    }

    console.log(`Cron found ${duePosts.length} due posts.`);

    for (const post of duePosts) {
      try {
        console.log(`Cron publishing post ${post.id}`);
        await publishScheduledPost(post);
      } catch (error) {
        console.error(
          `Failed scheduled post ${post.id}:`,
          error.response?.data || error.message
        );
      }
    }
  } catch (error) {
    console.error("Cron scheduler failed:", error.response?.data || error.message);
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
