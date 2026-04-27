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
import { randomUUID, createHmac, timingSafeEqual } from "crypto";
import { fileURLToPath } from "url";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { createClient } from "@supabase/supabase-js";

dotenv.config();

const app = express();

app.use(cors());
app.use(
  express.json({
    limit: "50mb",
    verify: (req, _res, buf) => {
      req.rawBody = buf;
    },
  }),
);
app.use(express.urlencoded({ extended: true, limit: "50mb" }));

const PORT = process.env.PORT || 3000;

const IG_GRAPH_HOST = process.env.GRAPH_HOST || "https://graph.instagram.com";
const META_GRAPH_HOST = process.env.META_GRAPH_HOST || "https://graph.facebook.com";
const GRAPH_VERSION = process.env.GRAPH_VERSION || process.env.META_GRAPH_VERSION || "v25.0";

const META_ACCESS_TOKEN = process.env.META_ACCESS_TOKEN;
const IG_USER_ID = process.env.IG_USER_ID;

const META_APP_SECRET = process.env.META_APP_SECRET;
const META_WEBHOOK_VERIFY_TOKEN = process.env.META_WEBHOOK_VERIFY_TOKEN;

const FACEBOOK_PAGE_ID = process.env.FACEBOOK_PAGE_ID;
const FACEBOOK_PAGE_ACCESS_TOKEN = process.env.FACEBOOK_PAGE_ACCESS_TOKEN;

const AUTO_REPLY_ENABLED = String(process.env.AUTO_REPLY_ENABLED || "false").toLowerCase() === "true";

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
const AI_BULK_CONCURRENCY = Number(process.env.AI_BULK_CONCURRENCY || 2);

const CLOUDINARY_CLOUD_NAME = process.env.CLOUDINARY_CLOUD_NAME;
const CLOUDINARY_API_KEY = process.env.CLOUDINARY_API_KEY;
const CLOUDINARY_API_SECRET = process.env.CLOUDINARY_API_SECRET;

const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;

cloudinary.config({
  cloud_name: CLOUDINARY_CLOUD_NAME,
  api_key: CLOUDINARY_API_KEY,
  api_secret: CLOUDINARY_API_SECRET,
  secure: true,
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
    fileSize: 250 * 1024 * 1024,
  },
});

app.use((req, _res, next) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.originalUrl}`);
  next();
});

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function requireSupabaseConfig() {
  if (!supabase) {
    throw new Error("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY.");
  }
}

function requireCloudinaryConfig() {
  if (!CLOUDINARY_CLOUD_NAME || !CLOUDINARY_API_KEY || !CLOUDINARY_API_SECRET) {
    throw new Error("Missing Cloudinary environment variables.");
  }
}

function requireMetaPublishConfig() {
  if (!META_ACCESS_TOKEN || !IG_USER_ID) {
    throw new Error("Missing META_ACCESS_TOKEN or IG_USER_ID.");
  }
}

function requirePageMessagingConfig() {
  if (!FACEBOOK_PAGE_ID || !FACEBOOK_PAGE_ACCESS_TOKEN) {
    throw new Error("Missing FACEBOOK_PAGE_ID or FACEBOOK_PAGE_ACCESS_TOKEN.");
  }
}

function requireGeminiConfig(apiKey = GEMINI_API_KEY) {
  if (!apiKey) {
    throw new Error("Missing GEMINI_API_KEY.");
  }
}

function metaUrl(pathValue) {
  const cleanPath = String(pathValue).startsWith("/")
    ? String(pathValue)
    : `/${pathValue}`;
  return `${META_GRAPH_HOST}/${GRAPH_VERSION}${cleanPath}`;
}

function igUrl(pathValue) {
  const cleanPath = String(pathValue).startsWith("/")
    ? String(pathValue)
    : `/${pathValue}`;
  return `${IG_GRAPH_HOST}/${GRAPH_VERSION}${cleanPath}`;
}

function cleanupFile(filePath) {
  try {
    if (filePath && fs.existsSync(filePath)) {
      fs.unlinkSync(filePath);
    }
  } catch {
    // ignore
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
    String(value).trim(),
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

function parseJsonLoose(text) {
  try {
    return JSON.parse(String(text).replace(/```json|```/g, "").trim());
  } catch {
    return null;
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
  return body?.[camelKey] ?? body?.[snakeKey];
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

  if (name.endsWith(".mp4") || name.endsWith(".mov") || name.endsWith(".m4v")) {
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

function extractBodySource(body) {
  return body?.post || body?.item || body?.data || body?.payload || body || {};
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
    media.id,
  );

  const imageUrl = firstDefined(
    source.imageUrl,
    source.image_url,
    media.imageUrl,
    media.image_url,
  );

  const videoUrl = firstDefined(
    source.videoUrl,
    source.video_url,
    media.videoUrl,
    media.video_url,
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
    videoUrl,
  );

  let mediaType = firstDefined(
    source.mediaType,
    source.media_type,
    media.mediaType,
    media.media_type,
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
    mediaType,
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
    source.time,
  );
}

function normalizeCaption(body) {
  const source = extractBodySource(body);
  return String(firstDefined(source.caption, source.finalText, source.final_text, "") || "");
}

function normalizeInputHashtags(body) {
  const source = extractBodySource(body);
  return normalizeHashtags(source.hashtags || source.tags || []);
}

function withAliases(row) {
  if (!row || typeof row !== "object") return row;

  return {
    ...row,
    mediaUrl: row.media_url ?? row.mediaUrl,
    imageUrl: row.image_url ?? row.imageUrl,
    videoUrl: row.video_url ?? row.videoUrl,
    mediaType: row.media_type ?? row.mediaType,
    createdAt: row.created_at ?? row.createdAt,
    updatedAt: row.updated_at ?? row.updatedAt,
    scheduledAt: row.scheduled_at ?? row.scheduledAt,
    publishedAt: row.published_at ?? row.publishedAt,
    mediaAssetId: row.media_asset_id ?? row.mediaAssetId,
    sourceMediaAssetId: row.source_media_asset_id ?? row.sourceMediaAssetId,
    isAiGenerated: row.is_ai_generated ?? row.isAiGenerated,
    isPublished: row.is_published ?? row.isPublished,
    isScheduled: row.is_scheduled ?? row.isScheduled,
  };
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
    timeout: 30000,
  });

  const mimeType =
    response.headers["content-type"] &&
    response.headers["content-type"].startsWith("image/")
      ? response.headers["content-type"]
      : "image/jpeg";

  return {
    inlineData: {
      mimeType,
      data: Buffer.from(response.data).toString("base64"),
    },
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
    format: "jpg",
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
    overwrite: false,
  });

  return result.secure_url;
}

async function uploadNormalizedImage(file) {
  const normalizedPath = path.join(tempDir, `${randomUUID()}.jpg`);

  await sharp(file.path)
    .rotate()
    .jpeg({
      quality: 92,
      mozjpeg: true,
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
    mimeType: "image/jpeg",
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
    mimeType: file.mimetype || "video/mp4",
  };
}

/**
 * INSTAGRAM PUBLISH
 */
async function pollMediaContainerStatus(containerId) {
  await sleep(3000);

  for (let attempt = 1; attempt <= 24; attempt += 1) {
    try {
      const response = await axios.get(igUrl(`/${containerId}`), {
        params: {
          fields: "status_code,status",
          access_token: META_ACCESS_TOKEN,
        },
        timeout: 30000,
      });

      const statusCode = response.data?.status_code;
      const status = response.data?.status;

      if (statusCode === "FINISHED") return response.data;

      if (statusCode === "ERROR") {
        throw new Error(`Meta media container failed: ${status || "Unknown error"}`);
      }

      await sleep(5000);
    } catch (error) {
      if (attempt === 24) throw error;
      await sleep(5000);
    }
  }

  throw new Error("Meta media container did not finish processing in time.");
}

async function publishToInstagram({ mediaUrl, imageUrl, videoUrl, mediaType, caption }) {
  requireMetaPublishConfig();

  const finalMediaUrl = mediaUrl || imageUrl || videoUrl;
  const finalMediaType = mediaType || detectMediaTypeFromUrl(finalMediaUrl);

  if (!finalMediaUrl || !isPublicHttpsUrl(finalMediaUrl)) {
    throw new Error("mediaUrl must be a public HTTPS URL.");
  }

  if (finalMediaType !== "image" && finalMediaType !== "video") {
    throw new Error("mediaType must be image or video.");
  }

  const params = {
    caption: caption || "",
    access_token: META_ACCESS_TOKEN,
  };

  if (finalMediaType === "image") {
    params.image_url = finalMediaUrl;
    params.media_type = "IMAGE";
  }

  if (finalMediaType === "video") {
    params.video_url = finalMediaUrl;
    params.media_type = "VIDEO";
  }

  const containerResponse = await axios.post(igUrl(`/${IG_USER_ID}/media`), null, {
    params,
    timeout: 60000,
  });

  const creationId = containerResponse.data?.id;

  if (!creationId) {
    throw new Error("Meta did not return creation_id.");
  }

  await pollMediaContainerStatus(creationId);

  let publishResponse = null;

  for (let attempt = 1; attempt <= 3; attempt += 1) {
    try {
      publishResponse = await axios.post(igUrl(`/${IG_USER_ID}/media_publish`), null, {
        params: {
          creation_id: creationId,
          access_token: META_ACCESS_TOKEN,
        },
        timeout: 60000,
      });
      break;
    } catch (error) {
      const metaCode = error.response?.data?.error?.code;

      if (metaCode === 9007 && attempt < 3) {
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
    mediaUrl: finalMediaUrl,
  };
}

/**
 * GEMINI / AI
 */
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
      "Write like a transformation post. Emphasize how the decor changes the feeling of the space.",
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
  hashtagCount,
}) {
  const presetInstruction = getPresetInstruction(captionPreset, customPrompt);
  const safeHashtagCount = Number.isFinite(Number(hashtagCount))
    ? Math.max(3, Math.min(25, Number(hashtagCount)))
    : DEFAULT_HASHTAG_COUNT;

  return `
You are a senior Instagram marketing copywriter for ${businessName}, a ${location}-based company specializing in premium artificial trees, artificial flowers, custom greenery, and luxury decor installations.

Analyze the image carefully before writing.

Identify what is visible:
- Product type.
- Colors.
- Space style.
- Strongest marketing angle.

Caption preset:
${captionPreset}

Preset instruction:
${presetInstruction}

Language:
${language}

Tone:
${tone}

CTA:
${cta}

Hashtag count:
${safeHashtagCount}

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
- Arabic must sound natural and premium.
- Use no more than ${safeHashtagCount} hashtags.
`;
}

function buildFixedAiEditRules({
  preserveProduct = true,
  keepPot = true,
  outputSize = "1080x1350",
  aspectRatio = "4:5",
  resolution = 1080,
  quality = "high",
}) {
  const rules = [
    "The result must be photorealistic and look like real professional photography.",
    "Keep natural perspective, believable scale, correct shadows, realistic lighting, and harmonious colors.",
    "No cartoon, no painting, no CGI look, no artificial AI artifacts.",
    "No text, no watermark.",
  ];

  if (preserveProduct) {
    rules.push(
      "Keep the original tree / plant / flower arrangement exactly unchanged.",
      "Do NOT change product shape, structure, trunk, branches, leaves, flowers, colors, density, height, width, proportions, angle, or realism.",
      "Preserve the original product identity exactly.",
    );
  }

  if (keepPot) {
    rules.push(
      "Keep the pot / planter / base exactly unchanged unless explicitly asked.",
      "Do NOT change pot shape, color, material, texture, size, or placement.",
    );
  }

  rules.push(
    "Change only the background, environment, surrounding decor, floor, walls, lighting, and atmosphere.",
    `Output aspect ratio: ${aspectRatio}.`,
    `Target output size: ${outputSize}.`,
    `Target resolution: ${resolution}.`,
    `Quality level: ${quality}.`,
  );

  return rules;
}

function getAiEditPresetInstruction(preset, customPrompt) {
  const cleanPreset = String(preset || "Luxury Interior").trim();

  if (cleanPreset === "Custom") {
    return customPrompt && String(customPrompt).trim()
      ? String(customPrompt).trim()
      : "Create a realistic premium background that enhances the product.";
  }

  const presets = {
    "Change Background":
      "Replace the background with a clean premium realistic environment that suits the product.",
    "Luxury Interior":
      "Place the product in a refined luxury interior with warm neutral walls, elegant flooring, soft natural light, and premium decor.",
    "Villa Entrance":
      "Place the product in a luxurious villa entrance with elegant architecture, premium flooring, soft lighting, and a welcoming high-end atmosphere.",
    "Staircase Decor":
      "Create a realistic under-stair or staircase decor scene with elegant walls, premium flooring, soft lighting, stones or greenery accents where appropriate.",
    "Minimal Modern Space":
      "Place the product in a minimal modern interior with clean lines, warm neutral colors, subtle luxury, and calm composition.",
    "Commercial / Mall Decor":
      "Place the product in a realistic commercial interior such as a mall, showroom, lobby, or retail space with polished finishes and professional lighting.",
    "Hotel Lobby Decor":
      "Place the product in a luxury hotel lobby with elegant materials, soft ambient lighting, premium furniture, and a high-end hospitality feel.",
    "Outdoor Courtyard":
      "Place the product in a realistic outdoor courtyard with natural light, architectural walls, stone or marble floor, and subtle landscape details.",
    "Neutral Studio Background":
      "Place the product on a clean neutral studio background with realistic soft shadows and premium product photography lighting.",
  };

  return presets[cleanPreset] || presets["Luxury Interior"];
}

function buildAiEditPrompt({
  preset = "Luxury Interior",
  customPrompt = "",
  editMode = "background_only",
  preserveProduct = true,
  keepPot = true,
  aspectRatio = "4:5",
  outputSize = "1080x1350",
  resolution = 1080,
  quality = "high",
  backgroundIntensity = "medium",
  userPrompt = "",
}) {
  const fixedRules = buildFixedAiEditRules({
    preserveProduct,
    keepPot,
    outputSize,
    aspectRatio,
    resolution,
    quality,
  });

  const presetInstruction = getAiEditPresetInstruction(preset, customPrompt);

  return `
Professional AI image edit request.

Edit preset:
${preset}

Edit mode:
${editMode}

Background intensity:
${backgroundIntensity}

Main scene instruction:
${userPrompt && String(userPrompt).trim() ? String(userPrompt).trim() : presetInstruction}

Fixed preservation rules:
${fixedRules.map((rule) => `- ${rule}`).join("\n")}

Scene realism rules:
- Match the original camera angle.
- Product must sit naturally in the new space.
- Scale must be believable.
- Lighting direction must be consistent.
- Shadows must be realistic and grounded.
- Colors must be harmonious and not oversaturated.
- Suitable for premium Instagram marketing.

Return only the edited image.
`;
}

async function generateTextWithGemini(prompt, modelName = GEMINI_TEXT_MODEL, apiKey = GEMINI_API_KEY) {
  requireGeminiConfig(apiKey);

  const genAI = new GoogleGenerativeAI(apiKey);
  const model = genAI.getGenerativeModel({ model: modelName });

  const result = await model.generateContent(prompt);
  return result.response.text();
}

async function generateVisionWithGemini({
  imageUrl,
  prompt,
  modelName = GEMINI_TEXT_MODEL,
  apiKey = GEMINI_API_KEY,
  fallbackTextOnly = false,
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
  fallbackTextOnly = false,
}) {
  const selectedProvider = normalizeProvider(provider);
  const selectedApiKey = getProviderApiKey({ provider: selectedProvider, apiKey });

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
    hashtagCount,
  });

  if (selectedProvider !== "gemini") {
    throw new Error("Caption generation currently supports Gemini in this backend.");
  }

  const text = await generateVisionWithGemini({
    imageUrl,
    prompt,
    modelName: model || GEMINI_TEXT_MODEL,
    apiKey: selectedApiKey,
    fallbackTextOnly,
  });

  const parsed = parseJsonLoose(text);

  if (parsed) {
    return {
      caption: parsed.caption || "",
      hashtags: normalizeHashtags(parsed.hashtags),
      alt_text: parsed.alt_text || "",
      detected_product: parsed.detected_product || "",
      visual_description: parsed.visual_description || "",
      marketing_angle: parsed.marketing_angle || "",
    };
  }

  return {
    caption: text,
    hashtags: [],
    alt_text: "",
    detected_product: "",
    visual_description: "",
    marketing_angle: "",
  };
}

async function generateEditPromptWithGemini({
  imageUrl,
  preset = "Luxury Interior",
  editMode = "background_only",
  preserveProduct = true,
  keepPot = true,
  language = "english",
  customPrompt = "",
  model = GEMINI_TEXT_MODEL,
  apiKey = GEMINI_API_KEY,
}) {
  const basePrompt = buildAiEditPrompt({
    preset,
    customPrompt,
    editMode,
    preserveProduct,
    keepPot,
  });

  const analysisPrompt = `
Analyze this product image and create a professional AI image editing prompt.

Language:
${language}

Preset:
${preset}

Edit mode:
${editMode}

Mandatory fixed rules:
${basePrompt}

Return strict JSON only:
{
  "prompt": "full detailed edit prompt",
  "preset": "${preset}",
  "editMode": "${editMode}",
  "fixedRules": ["rule 1", "rule 2"]
}
`;

  const text = await generateVisionWithGemini({
    imageUrl,
    prompt: analysisPrompt,
    modelName: model,
    apiKey,
    fallbackTextOnly: false,
  });

  const parsed = parseJsonLoose(text);

  return {
    prompt: parsed?.prompt || String(text).replace(/```json|```/g, "").trim(),
    preset: parsed?.preset || preset,
    editMode: parsed?.editMode || editMode,
    fixedRules:
      Array.isArray(parsed?.fixedRules) && parsed.fixedRules.length > 0
        ? parsed.fixedRules
        : buildFixedAiEditRules({}),
  };
}

async function editImageWithGemini({ originalImageUrl, prompt, model = GEMINI_IMAGE_MODEL }) {
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
        parts: [{ text: prompt }, imagePart],
      },
    ],
  };

  const response = await axios.post(
    `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`,
    requestBody,
    {
      params: { key: GEMINI_API_KEY },
      headers: { "Content-Type": "application/json" },
      timeout: 180000,
    },
  );

  const parts = response.data?.candidates?.[0]?.content?.parts || [];
  const imageOutput = parts.find((part) => part.inlineData || part.inline_data);

  if (!imageOutput) {
    const textOutput = parts.map((part) => part.text).filter(Boolean).join("\n");

    throw new Error(
      textOutput ||
        "Gemini image model did not return image data. Check image model access.",
    );
  }

  const inlineData = imageOutput.inlineData || imageOutput.inline_data;

  const tempImagePath = saveBase64TempImage({
    base64Data: inlineData.data,
    mimeType: inlineData.mimeType || inlineData.mime_type || "image/png",
  });

  const normalizedPath = path.join(tempDir, `${randomUUID()}.jpg`);

  await sharp(tempImagePath)
    .rotate()
    .jpeg({ quality: 92, mozjpeg: true })
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
    model,
  };
}

async function insertAiGeneratedMedia({
  editedImageUrl,
  sourceMediaAssetId,
  aiJobId,
  prompt,
  editMode,
  provider,
  model,
  aspectRatio,
  outputSize,
  resolution,
  quality,
}) {
  requireSupabaseConfig();

  const { data, error } = await supabase
    .from("media_assets")
    .insert({
      media_url: editedImageUrl,
      image_url: editedImageUrl,
      video_url: null,
      media_type: "image",
      mime_type: "image/jpeg",
      file_name: "ai-edited-image.jpg",
      is_uploaded: true,
      is_scheduled: false,
      is_published: false,
      source_media_asset_id: isValidUuid(sourceMediaAssetId) ? sourceMediaAssetId : null,
      is_ai_generated: true,
      ai_job_id: isValidUuid(aiJobId) ? aiJobId : null,
      ai_prompt: prompt,
      ai_edit_mode: editMode,
      ai_provider: provider,
      ai_model: model,
      ai_aspect_ratio: aspectRatio,
      ai_output_size: outputSize,
      ai_resolution: Number(resolution) || null,
      ai_quality: quality,
    })
    .select()
    .single();

  if (error) throw error;

  return data;
}

/**
 * SCHEDULE HELPERS
 */
async function markMediaPublished(mediaAssetId, publishedAt) {
  if (!mediaAssetId || !isValidUuid(mediaAssetId)) return;

  requireSupabaseConfig();

  await supabase
    .from("media_assets")
    .update({
      is_published: true,
      published_at: publishedAt,
      updated_at: new Date().toISOString(),
    })
    .eq("id", mediaAssetId);
}

async function publishScheduledPost(post) {
  requireSupabaseConfig();

  await supabase
    .from("scheduled_posts")
    .update({
      status: "publishing",
      publish_attempts: (post.publish_attempts || 0) + 1,
      updated_at: new Date().toISOString(),
    })
    .eq("id", post.id);

  try {
    const result = await publishToInstagram({
      mediaUrl: post.media_url,
      imageUrl: post.image_url,
      videoUrl: post.video_url,
      mediaType: post.media_type,
      caption: post.final_text || post.caption || "",
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
        updated_at: publishedAt,
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
        updated_at: new Date().toISOString(),
      })
      .eq("id", post.id);

    throw error;
  }
}

async function createScheduledPostFromInput(inputBody) {
  requireSupabaseConfig();

  let { mediaAssetId, mediaUrl, imageUrl, videoUrl, mediaType } =
    normalizeMediaInput(inputBody);

  const mediaAsset = await resolveMediaAsset({ mediaAssetId, mediaUrl });

  if (mediaAsset) {
    mediaAssetId = mediaAsset.id;
    mediaUrl = mediaUrl || mediaAsset.media_url;
    imageUrl = imageUrl || mediaAsset.image_url;
    videoUrl = videoUrl || mediaAsset.video_url;
    mediaType = mediaType || mediaAsset.media_type;
  } else if (!isValidUuid(mediaAssetId)) {
    mediaAssetId = null;
  }

  const caption = normalizeCaption(inputBody);
  const hashtags = normalizeInputHashtags(inputBody);
  const scheduledAt = normalizeScheduledAt(inputBody);

  if (!mediaUrl || !scheduledAt) {
    const error = new Error("mediaUrl and scheduledAt are required.");
    error.statusCode = 400;
    error.details = { received: inputBody };
    throw error;
  }

  if (mediaType !== "image" && mediaType !== "video") {
    const error = new Error("mediaType must be image or video.");
    error.statusCode = 400;
    error.details = { received: inputBody };
    throw error;
  }

  const parsedDate = new Date(scheduledAt);

  if (Number.isNaN(parsedDate.getTime())) {
    const error = new Error("scheduledAt is not a valid date.");
    error.statusCode = 400;
    throw error;
  }

  const finalText = `${caption}\n${hashtags.join(" ")}`.trim();

  const { data: post, error } = await supabase
    .from("scheduled_posts")
    .insert({
      media_asset_id: mediaAssetId,
      media_url: mediaUrl,
      image_url: imageUrl,
      video_url: videoUrl,
      media_type: mediaType,
      caption,
      hashtags,
      final_text: finalText,
      scheduled_at: parsedDate.toISOString(),
      status: "approved",
    })
    .select()
    .single();

  if (error) throw error;

  if (mediaAssetId && isValidUuid(mediaAssetId)) {
    await supabase
      .from("media_assets")
      .update({
        is_scheduled: true,
        scheduled_post_id: post.id,
        updated_at: new Date().toISOString(),
      })
      .eq("id", mediaAssetId);
  }

  return post;
}

async function runWithConcurrency(items, concurrency, task) {
  if (!items || items.length === 0) return;

  let nextIndex = 0;
  const safeConcurrency = Math.max(1, Math.min(Number(concurrency) || 1, items.length));

  async function worker() {
    while (true) {
      const currentIndex = nextIndex;
      nextIndex += 1;
      if (currentIndex >= items.length) return;
      await task(items[currentIndex], currentIndex);
    }
  }

  await Promise.all(Array.from({ length: safeConcurrency }, () => worker()));
}

async function processAiBulkJob(jobId) {
  requireSupabaseConfig();

  const { data: job, error: jobError } = await supabase
    .from("ai_jobs")
    .select("*")
    .eq("id", jobId)
    .single();

  if (jobError) {
    console.error("Could not load AI job:", jobError);
    return;
  }

  await supabase.from("ai_jobs").update({ status: "processing" }).eq("id", jobId);

  const { data: items, error: itemsError } = await supabase
    .from("ai_job_items")
    .select("*")
    .eq("ai_job_id", jobId)
    .order("created_at", { ascending: true });

  if (itemsError) {
    await supabase
      .from("ai_jobs")
      .update({ status: "failed", error_message: itemsError.message })
      .eq("id", jobId);
    return;
  }

  let completed = 0;
  let failed = 0;

  await runWithConcurrency(items || [], AI_BULK_CONCURRENCY, async (item) => {
    await supabase.from("ai_job_items").update({ status: "processing" }).eq("id", item.id);

    try {
      const result = await editImageWithGemini({
        originalImageUrl: item.original_image_url,
        prompt: item.prompt,
        model: job.model || GEMINI_IMAGE_MODEL,
      });

      const savedMedia = await insertAiGeneratedMedia({
        editedImageUrl: result.editedImageUrl,
        sourceMediaAssetId: item.original_media_asset_id,
        aiJobId: job.id,
        prompt: item.prompt,
        editMode: job.edit_mode,
        provider: job.provider,
        model: job.model,
        aspectRatio: job.aspect_ratio,
        outputSize: job.output_size,
        resolution: job.resolution,
        quality: job.quality,
      });

      await supabase
        .from("ai_job_items")
        .update({
          status: "completed",
          edited_image_url: result.editedImageUrl,
          result_media_asset_id: savedMedia?.id || null,
          error_message: null,
        })
        .eq("id", item.id);

      completed += 1;
    } catch (error) {
      failed += 1;

      await supabase
        .from("ai_job_items")
        .update({
          status: "failed",
          error_message: error.response?.data
            ? JSON.stringify(error.response.data)
            : error.message,
        })
        .eq("id", item.id);
    }

    await supabase
      .from("ai_jobs")
      .update({
        completed_items: completed,
        failed_items: failed,
      })
      .eq("id", jobId);
  });

  await supabase
    .from("ai_jobs")
    .update({
      status: failed > 0 && completed === 0 ? "failed" : "completed",
      completed_items: completed,
      failed_items: failed,
    })
    .eq("id", jobId);
}

/**
 * META WEBHOOK HELPERS
 */
function verifyMetaSignature(req) {
  if (!META_APP_SECRET) return true;

  const signature = req.headers["x-hub-signature-256"];

  if (!signature || !signature.startsWith("sha256=")) {
    return false;
  }

  const expected = `sha256=${createHmac("sha256", META_APP_SECRET)
    .update(req.rawBody || Buffer.from(""))
    .digest("hex")}`;

  const sigBuffer = Buffer.from(signature);
  const expectedBuffer = Buffer.from(expected);

  if (sigBuffer.length !== expectedBuffer.length) return false;

  return timingSafeEqual(sigBuffer, expectedBuffer);
}

async function storeWebhookEvent({ body, field = null, eventType = null }) {
  if (!supabase) return null;

  const { data, error } = await supabase
    .from("meta_webhook_events")
    .insert({
      object_type: body?.object || null,
      field,
      event_type: eventType,
      meta_id: null,
      payload: body,
      processed: false,
    })
    .select()
    .single();

  if (error) {
    console.warn("Webhook event insert failed:", error.message);
    return null;
  }

  return data;
}

async function upsertConversationFromMessage({ senderId, text, sentAt, raw }) {
  requireSupabaseConfig();

  if (!senderId) return null;

  const { data: existing } = await supabase
    .from("ig_conversations")
    .select("*")
    .eq("ig_scoped_user_id", senderId)
    .maybeSingle();

  if (existing) {
    const { data, error } = await supabase
      .from("ig_conversations")
      .update({
        last_message_text: text || existing.last_message_text,
        last_message_at: sentAt || new Date().toISOString(),
        unread_count: (existing.unread_count || 0) + 1,
        updated_at: new Date().toISOString(),
      })
      .eq("id", existing.id)
      .select()
      .single();

    if (error) throw error;
    return data;
  }

  const { data, error } = await supabase
    .from("ig_conversations")
    .insert({
      platform: "instagram",
      ig_scoped_user_id: senderId,
      username: senderId,
      last_message_text: text || "",
      last_message_at: sentAt || new Date().toISOString(),
      unread_count: 1,
      status: "open",
      raw,
    })
    .select()
    .single();

  if (error) throw error;
  return data;
}

async function insertInboundMessageFromWebhook(messagingEvent) {
  requireSupabaseConfig();

  const senderId = messagingEvent?.sender?.id;
  const recipientId = messagingEvent?.recipient?.id;
  const message = messagingEvent?.message || {};
  const text = message.text || "";
  const mid = message.mid || null;
  const timestamp = messagingEvent?.timestamp
    ? new Date(Number(messagingEvent.timestamp)).toISOString()
    : new Date().toISOString();

  const attachment = Array.isArray(message.attachments) ? message.attachments[0] : null;
  const messageType = attachment?.type || (text ? "text" : "unknown");
  const mediaUrl = attachment?.payload?.url || null;

  const conversation = await upsertConversationFromMessage({
    senderId,
    text: text || messageType,
    sentAt: timestamp,
    raw: messagingEvent,
  });

  if (!conversation) return;

  const { error } = await supabase.from("ig_messages").upsert(
    {
      conversation_id: conversation.id,
      meta_message_id: mid,
      ig_scoped_user_id: senderId,
      direction: "inbound",
      message_type: messageType,
      text,
      media_url: mediaUrl,
      from_id: senderId,
      to_id: recipientId,
      sent_at: timestamp,
      raw: messagingEvent,
    },
    { onConflict: "meta_message_id" },
  );

  if (error) {
    console.warn("Inbound message insert failed:", error.message);
  }
}

function autoReplyRuleMatches(rule, commentText) {
  const text = String(commentText || "").toLowerCase().trim();

  if (!rule.is_enabled) return false;

  if (rule.trigger_type === "any_comment") return true;

  const keywords = Array.isArray(rule.keywords)
    ? rule.keywords.map((k) => String(k).toLowerCase().trim()).filter(Boolean)
    : [];

  if (keywords.length === 0) return false;

  if (rule.trigger_type === "keyword") {
    return keywords.some((keyword) => text.includes(keyword));
  }

  if (rule.trigger_type === "exact_match") {
    return keywords.some((keyword) => text === keyword);
  }

  return false;
}

async function sendCommentPublicReply(igCommentId, message) {
  const token = META_ACCESS_TOKEN || FACEBOOK_PAGE_ACCESS_TOKEN;

  const response = await axios.post(metaUrl(`/${igCommentId}/replies`), null, {
    params: {
      message,
      access_token: token,
    },
    timeout: 30000,
  });

  return response.data;
}

async function sendCommentPrivateReply(igCommentId, message) {
  const token = META_ACCESS_TOKEN || FACEBOOK_PAGE_ACCESS_TOKEN;

  const response = await axios.post(metaUrl(`/${igCommentId}/private_replies`), null, {
    params: {
      message,
      access_token: token,
    },
    timeout: 30000,
  });

  return response.data;
}

async function applyAutoReplyRulesToComment(commentRow) {
  if (!AUTO_REPLY_ENABLED || !commentRow?.ig_comment_id) return;

  requireSupabaseConfig();

  const { data: rules, error } = await supabase
    .from("ig_auto_reply_rules")
    .select("*")
    .eq("is_enabled", true);

  if (error) {
    console.warn("Could not load auto reply rules:", error.message);
    return;
  }

  for (const rule of rules || []) {
    if (!autoReplyRuleMatches(rule, commentRow.text)) continue;

    if (rule.only_once_per_user && commentRow.user_id) {
      const { data: existingLog } = await supabase
        .from("ig_auto_reply_logs")
        .select("id")
        .eq("rule_id", rule.id)
        .eq("user_id", commentRow.user_id)
        .eq("status", "sent")
        .maybeSingle();

      if (existingLog) {
        await supabase.from("ig_auto_reply_logs").insert({
          rule_id: rule.id,
          ig_comment_id: commentRow.ig_comment_id,
          ig_media_id: commentRow.ig_media_id,
          username: commentRow.username,
          user_id: commentRow.user_id,
          reply_mode: rule.reply_mode,
          status: "skipped",
          message_text: "Skipped because only_once_per_user is enabled.",
        });
        continue;
      }
    }

    try {
      if ((rule.reply_mode === "public_reply" || rule.reply_mode === "both") && rule.public_reply_text) {
        await sendCommentPublicReply(commentRow.ig_comment_id, rule.public_reply_text);
      }

      if ((rule.reply_mode === "private_reply" || rule.reply_mode === "both") && rule.private_reply_text) {
        await sendCommentPrivateReply(commentRow.ig_comment_id, rule.private_reply_text);
      }

      await supabase.from("ig_auto_reply_logs").insert({
        rule_id: rule.id,
        ig_comment_id: commentRow.ig_comment_id,
        ig_media_id: commentRow.ig_media_id,
        username: commentRow.username,
        user_id: commentRow.user_id,
        reply_mode: rule.reply_mode,
        message_text:
          rule.reply_mode === "public_reply"
            ? rule.public_reply_text
            : rule.private_reply_text || rule.public_reply_text,
        status: "sent",
      });

      await supabase
        .from("ig_comments")
        .update({
          auto_reply_rule_id: rule.id,
          replied_by_app: rule.reply_mode === "public_reply" || rule.reply_mode === "both",
          replied_at:
            rule.reply_mode === "public_reply" || rule.reply_mode === "both"
              ? new Date().toISOString()
              : commentRow.replied_at,
          private_replied_by_app:
            rule.reply_mode === "private_reply" || rule.reply_mode === "both",
          private_replied_at:
            rule.reply_mode === "private_reply" || rule.reply_mode === "both"
              ? new Date().toISOString()
              : commentRow.private_replied_at,
        })
        .eq("ig_comment_id", commentRow.ig_comment_id);
    } catch (replyError) {
      await supabase.from("ig_auto_reply_logs").insert({
        rule_id: rule.id,
        ig_comment_id: commentRow.ig_comment_id,
        ig_media_id: commentRow.ig_media_id,
        username: commentRow.username,
        user_id: commentRow.user_id,
        reply_mode: rule.reply_mode,
        message_text:
          rule.reply_mode === "public_reply"
            ? rule.public_reply_text
            : rule.private_reply_text || rule.public_reply_text,
        status: "failed",
        error_message: replyError.response?.data
          ? JSON.stringify(replyError.response.data)
          : replyError.message,
      });
    }
  }
}

async function upsertCommentFromMeta(value) {
  requireSupabaseConfig();

  const igCommentId = value.id || value.comment_id;
  if (!igCommentId) return null;

  const row = {
    ig_comment_id: igCommentId,
    ig_media_id: value.media_id || value.media?.id || null,
    parent_comment_id: value.parent_id || value.parent_comment_id || null,
    username: value.from?.username || value.username || null,
    user_id: value.from?.id || value.user_id || null,
    text: value.text || value.message || "",
    like_count: Number(value.like_count || 0),
    timestamp: value.timestamp ? new Date(value.timestamp).toISOString() : new Date().toISOString(),
    is_reply: Boolean(value.parent_id || value.parent_comment_id),
    raw: value,
  };

  const { data, error } = await supabase
    .from("ig_comments")
    .upsert(row, { onConflict: "ig_comment_id" })
    .select()
    .single();

  if (error) {
    console.warn("Comment upsert failed:", error.message);
    return null;
  }

  await applyAutoReplyRulesToComment(data);
  return data;
}

/**
 * ROOT / HEALTH
 */
app.get("/", (_req, res) => {
  res.json({
    ok: true,
    app: "AutoFlow Backend",
    status: "running",
    graphVersion: GRAPH_VERSION,
    cloudinaryConfigured: Boolean(CLOUDINARY_CLOUD_NAME && CLOUDINARY_API_KEY && CLOUDINARY_API_SECRET),
    supabaseConfigured: Boolean(supabase),
    aiStudio: true,
    inboxEnabled: Boolean(FACEBOOK_PAGE_ID && FACEBOOK_PAGE_ACCESS_TOKEN),
    commentsEnabled: Boolean(META_ACCESS_TOKEN || FACEBOOK_PAGE_ACCESS_TOKEN),
    webhooksEnabled: Boolean(META_WEBHOOK_VERIFY_TOKEN),
  });
});

app.get("/api/health", (_req, res) => {
  res.json({
    ok: true,
    timestamp: new Date().toISOString(),
    graphVersion: GRAPH_VERSION,
    cloudinaryConfigured: Boolean(CLOUDINARY_CLOUD_NAME && CLOUDINARY_API_KEY && CLOUDINARY_API_SECRET),
    supabaseConfigured: Boolean(supabase),
    aiStudio: true,
    inboxEnabled: Boolean(FACEBOOK_PAGE_ID && FACEBOOK_PAGE_ACCESS_TOKEN),
    commentsEnabled: Boolean(META_ACCESS_TOKEN || FACEBOOK_PAGE_ACCESS_TOKEN),
    webhooksEnabled: Boolean(META_WEBHOOK_VERIFY_TOKEN),
  });
});

/**
 * WEBHOOKS
 */
app.get("/api/webhooks/meta", (req, res) => {
  const mode = req.query["hub.mode"];
  const token = req.query["hub.verify_token"];
  const challenge = req.query["hub.challenge"];

  if (mode === "subscribe" && token === META_WEBHOOK_VERIFY_TOKEN) {
    return res.status(200).send(challenge);
  }

  return res.status(403).send("Invalid verify token");
});

app.post("/api/webhooks/meta", async (req, res) => {
  try {
    if (!verifyMetaSignature(req)) {
      return res.status(403).json({
        ok: false,
        error: "Invalid Meta signature.",
      });
    }

    const body = req.body;
    const event = await storeWebhookEvent({ body, field: null, eventType: "webhook" });

    for (const entry of body.entry || []) {
      for (const messagingEvent of entry.messaging || []) {
        await insertInboundMessageFromWebhook(messagingEvent);
      }

      for (const change of entry.changes || []) {
        if (change.field && String(change.field).includes("comment")) {
          await upsertCommentFromMeta(change.value || {});
        }

        if (change.field && String(change.field).includes("message")) {
          await storeWebhookEvent({
            body: change,
            field: change.field,
            eventType: "message_change",
          });
        }
      }
    }

    if (event?.id) {
      await supabase
        ?.from("meta_webhook_events")
        .update({ processed: true })
        .eq("id", event.id);
    }

    return res.json({ ok: true });
  } catch (error) {
    console.error("Webhook processing failed:", error.response?.data || error.message);
    return res.status(500).json({
      ok: false,
      error: error.response?.data || error.message,
    });
  }
});

/**
 * META TEST
 */
async function testMetaConnection() {
  requireMetaPublishConfig();

  const response = await axios.get(igUrl("/me"), {
    params: {
      fields: "user_id,username",
      access_token: META_ACCESS_TOKEN,
    },
    timeout: 30000,
  });

  return {
    account: response.data,
    configuredIgUserId: IG_USER_ID,
    idMatches: String(response.data.user_id) === String(IG_USER_ID),
    graphHost: IG_GRAPH_HOST,
    graphVersion: GRAPH_VERSION,
  };
}

app.get("/api/meta/test-connection", async (_req, res) => {
  try {
    const result = await testMetaConnection();
    res.json({ ok: true, ...result });
  } catch (error) {
    res.status(500).json({
      ok: false,
      error: error.response?.data || error.message,
    });
  }
});

app.post("/api/meta/test-connection", async (_req, res) => {
  try {
    const result = await testMetaConnection();
    res.json({ ok: true, ...result });
  } catch (error) {
    res.status(500).json({
      ok: false,
      error: error.response?.data || error.message,
    });
  }
});

/**
 * MEDIA
 */
app.post("/api/upload", upload.any(), async (req, res) => {
  try {
    requireCloudinaryConfig();
    requireSupabaseConfig();

    const file = req.files?.[0];

    if (!file) {
      return res.status(400).json({
        ok: false,
        error: "No media file uploaded.",
      });
    }

    const mediaType = detectMediaTypeFromFile(file);

    if (!mediaType) {
      cleanupFile(file.path);
      return res.status(400).json({
        ok: false,
        error: "Unsupported file type. Upload image or video only.",
      });
    }

    const uploaded = mediaType === "image" ? await uploadNormalizedImage(file) : await uploadVideo(file);

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
        is_published: false,
      })
      .select()
      .single();

    if (error) throw error;

    res.json({
      ok: true,
      media: withAliases(media),
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
      mime_type: uploaded.mimeType,
    });
  } catch (error) {
    console.error("Upload failed:", error.response?.data || error.message);

    if (req.files) {
      for (const file of req.files) cleanupFile(file.path);
    }

    res.status(500).json({
      ok: false,
      error: error.response?.data || error.message,
    });
  }
});

app.get("/api/media", async (_req, res) => {
  try {
    requireSupabaseConfig();

    const { data, error } = await supabase
      .from("media_assets")
      .select("*")
      .order("created_at", { ascending: false });

    if (error) throw error;

    res.json({
      ok: true,
      media: (data || []).map(withAliases),
    });
  } catch (error) {
    res.status(500).json({ ok: false, error: error.message });
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

    res.json({ ok: true, deleted: true });
  } catch (error) {
    res.status(500).json({ ok: false, error: error.message });
  }
});

/**
 * PUBLISH NOW
 */
app.post("/api/meta/publish-now", async (req, res) => {
  try {
    const { mediaAssetId, mediaUrl, imageUrl, videoUrl, mediaType } =
      normalizeMediaInput(req.body);

    const caption = req.body.caption || req.body.final_text || req.body.finalText || "";

    const result = await publishToInstagram({
      mediaUrl,
      imageUrl,
      videoUrl,
      mediaType,
      caption,
    });

    if (mediaAssetId && isValidUuid(mediaAssetId)) {
      await markMediaPublished(mediaAssetId, new Date().toISOString());
    }

    res.json({ ok: true, result });
  } catch (error) {
    res.status(500).json({
      ok: false,
      error: error.response?.data || error.message,
    });
  }
});

/**
 * CAPTION
 */
app.post("/api/gemini/generate-caption", async (req, res) => {
  try {
    const imageUrl =
      getBodyValue(req.body, "imageUrl", "image_url") ||
      getBodyValue(req.body, "mediaUrl", "media_url");

    if (!imageUrl) {
      return res.status(400).json({
        ok: false,
        error: "imageUrl or mediaUrl is required.",
      });
    }

    const result = await generateCaptionWithAI({
      provider: req.body.provider || "gemini",
      apiKey: req.body.apiKey,
      imageUrl,
      language: req.body.language || "arabic",
      tone: req.body.tone || "premium",
      model: req.body.model || GEMINI_TEXT_MODEL,
      captionPreset:
        req.body.captionPreset ||
        req.body.caption_preset ||
        "Luxury Product Caption",
      customPrompt: req.body.customPrompt || req.body.custom_prompt || "",
      businessName:
        req.body.businessName ||
        req.body.business_name ||
        DEFAULT_BUSINESS_NAME,
      location: req.body.location || DEFAULT_LOCATION,
      cta: req.body.cta || DEFAULT_CTA,
      hashtagCount:
        req.body.hashtagCount ||
        req.body.hashtag_count ||
        DEFAULT_HASHTAG_COUNT,
      fallbackTextOnly: req.body.fallbackTextOnly === true,
    });

    const mediaAssetId = req.body.mediaAssetId || req.body.media_asset_id;

    if (isValidUuid(mediaAssetId)) {
      await supabase
        .from("media_assets")
        .update({
          caption: result.caption,
          hashtags: result.hashtags,
          updated_at: new Date().toISOString(),
        })
        .eq("id", mediaAssetId);
    }

    res.json({ ok: true, result });
  } catch (error) {
    console.error("Generate caption failed:", error.response?.data || error.message);

    res.status(500).json({
      ok: false,
      error: error.response?.data || error.message,
    });
  }
});

/**
 * AI STUDIO
 */
app.post("/api/ai/generate-edit-prompt", async (req, res) => {
  try {
    const imageUrl =
      getBodyValue(req.body, "imageUrl", "image_url") ||
      getBodyValue(req.body, "mediaUrl", "media_url");

    if (!imageUrl) {
      return res.status(400).json({
        ok: false,
        error: "imageUrl or mediaUrl is required.",
      });
    }

    const result = await generateEditPromptWithGemini({
      imageUrl,
      preset: req.body.preset || "Luxury Interior",
      editMode: req.body.editMode || req.body.edit_mode || "background_only",
      preserveProduct: req.body.preserveProduct !== false,
      keepPot: req.body.keepPot !== false,
      language: req.body.language || "english",
      customPrompt: req.body.customPrompt || req.body.custom_prompt || "",
      model: req.body.model || GEMINI_TEXT_MODEL,
      apiKey: GEMINI_API_KEY,
    });

    res.json({ ok: true, result });
  } catch (error) {
    res.status(500).json({
      ok: false,
      error: error.response?.data || error.message,
    });
  }
});

app.post("/api/gemini/generate-edit-prompt", async (req, res) => {
  req.url = "/api/ai/generate-edit-prompt";
  app.handle(req, res);
});

app.post("/api/ai/edit-image", async (req, res) => {
  try {
    requireSupabaseConfig();

    const originalImageUrl =
      getBodyValue(req.body, "originalImageUrl", "original_image_url") ||
      getBodyValue(req.body, "imageUrl", "image_url") ||
      getBodyValue(req.body, "mediaUrl", "media_url");

    if (!originalImageUrl) {
      return res.status(400).json({
        ok: false,
        error: "originalImageUrl is required.",
      });
    }

    const model = req.body.model || GEMINI_IMAGE_MODEL;
    const mediaAssetId = req.body.mediaAssetId || req.body.media_asset_id || null;
    const editMode = req.body.editMode || req.body.edit_mode || "background_only";
    const aspectRatio = req.body.aspectRatio || req.body.aspect_ratio || "4:5";
    const outputSize = req.body.outputSize || req.body.output_size || "1080x1350";
    const resolution = Number(req.body.resolution || 1080);
    const quality = req.body.quality || "high";
    const backgroundIntensity =
      req.body.backgroundIntensity ||
      req.body.background_intensity ||
      "medium";
    const saveToLibrary = req.body.saveToLibrary !== false;
    const userPrompt = req.body.prompt || req.body.ai_edit_prompt || "";

    if (!userPrompt) {
      return res.status(400).json({ ok: false, error: "prompt is required." });
    }

    const finalPrompt = buildAiEditPrompt({
      preset: req.body.preset || "Custom",
      customPrompt: req.body.customPrompt || "",
      editMode,
      preserveProduct: req.body.preserveProduct !== false,
      keepPot: req.body.keepPot !== false,
      aspectRatio,
      outputSize,
      resolution,
      quality,
      backgroundIntensity,
      userPrompt,
    });

    const { data: job, error: jobError } = await supabase
      .from("ai_jobs")
      .insert({
        job_type: "single_edit",
        status: "processing",
        total_items: 1,
        completed_items: 0,
        failed_items: 0,
        provider: "gemini",
        model,
        edit_mode: editMode,
        aspect_ratio: aspectRatio,
        output_size: outputSize,
        resolution,
        quality,
      })
      .select()
      .single();

    if (jobError) throw jobError;

    const { data: jobItem, error: itemError } = await supabase
      .from("ai_job_items")
      .insert({
        ai_job_id: job.id,
        original_media_asset_id: isValidUuid(mediaAssetId) ? mediaAssetId : null,
        original_image_url: originalImageUrl,
        prompt: finalPrompt,
        status: "processing",
      })
      .select()
      .single();

    if (itemError) throw itemError;

    try {
      const result = await editImageWithGemini({
        originalImageUrl,
        prompt: finalPrompt,
        model,
      });

      let savedMedia = null;

      if (saveToLibrary) {
        savedMedia = await insertAiGeneratedMedia({
          editedImageUrl: result.editedImageUrl,
          sourceMediaAssetId: mediaAssetId,
          aiJobId: job.id,
          prompt: finalPrompt,
          editMode,
          provider: "gemini",
          model,
          aspectRatio,
          outputSize,
          resolution,
          quality,
        });
      }

      await supabase
        .from("ai_job_items")
        .update({
          status: "completed",
          edited_image_url: result.editedImageUrl,
          result_media_asset_id: savedMedia?.id || null,
        })
        .eq("id", jobItem.id);

      await supabase
        .from("ai_jobs")
        .update({
          status: "completed",
          completed_items: 1,
          failed_items: 0,
        })
        .eq("id", job.id);

      return res.json({
        ok: true,
        result: {
          jobId: job.id,
          originalMediaAssetId: mediaAssetId,
          editedImageUrl: result.editedImageUrl,
          prompt: finalPrompt,
          status: "completed",
          savedMediaAssetId: savedMedia?.id || null,
          media: savedMedia ? withAliases(savedMedia) : null,
        },
      });
    } catch (error) {
      await supabase
        .from("ai_job_items")
        .update({
          status: "failed",
          error_message: error.response?.data
            ? JSON.stringify(error.response.data)
            : error.message,
        })
        .eq("id", jobItem.id);

      await supabase
        .from("ai_jobs")
        .update({
          status: "failed",
          failed_items: 1,
          error_message: error.response?.data
            ? JSON.stringify(error.response.data)
            : error.message,
        })
        .eq("id", job.id);

      throw error;
    }
  } catch (error) {
    res.status(500).json({
      ok: false,
      error: error.response?.data || error.message,
    });
  }
});

app.post("/api/ai/bulk-edit", async (req, res) => {
  try {
    requireSupabaseConfig();

    const items = Array.isArray(req.body.items) ? req.body.items : [];
    const commonOptions = req.body.commonOptions || req.body.common_options || {};

    if (items.length === 0) {
      return res.status(400).json({
        ok: false,
        error: "items must be a non-empty array.",
      });
    }

    const model = commonOptions.model || req.body.model || GEMINI_IMAGE_MODEL;
    const editMode = commonOptions.editMode || commonOptions.edit_mode || "background_only";
    const aspectRatio = commonOptions.aspectRatio || commonOptions.aspect_ratio || "4:5";
    const outputSize = commonOptions.outputSize || commonOptions.output_size || "1080x1350";
    const resolution = Number(commonOptions.resolution || 1080);
    const quality = commonOptions.quality || "high";

    const { data: job, error: jobError } = await supabase
      .from("ai_jobs")
      .insert({
        job_type: "bulk_edit",
        status: "pending",
        total_items: items.length,
        completed_items: 0,
        failed_items: 0,
        provider: "gemini",
        model,
        edit_mode: editMode,
        aspect_ratio: aspectRatio,
        output_size: outputSize,
        resolution,
        quality,
      })
      .select()
      .single();

    if (jobError) throw jobError;

    const jobItemsPayload = items.map((item) => {
      const originalImageUrl =
        item.originalImageUrl ||
        item.original_image_url ||
        item.imageUrl ||
        item.image_url ||
        item.mediaUrl ||
        item.media_url;

      const mediaAssetId = item.mediaAssetId || item.media_asset_id || null;
      const rawPrompt = item.prompt || commonOptions.prompt || "";

      const finalPrompt = buildAiEditPrompt({
        preset: item.preset || commonOptions.preset || "Custom",
        customPrompt: item.customPrompt || commonOptions.customPrompt || "",
        editMode,
        preserveProduct: item.preserveProduct !== false,
        keepPot: item.keepPot !== false,
        aspectRatio,
        outputSize,
        resolution,
        quality,
        backgroundIntensity:
          item.backgroundIntensity ||
          commonOptions.backgroundIntensity ||
          "medium",
        userPrompt: rawPrompt,
      });

      return {
        ai_job_id: job.id,
        original_media_asset_id: isValidUuid(mediaAssetId) ? mediaAssetId : null,
        original_image_url: originalImageUrl,
        prompt: finalPrompt,
        status: "pending",
      };
    });

    const invalid = jobItemsPayload.find((item) => !item.original_image_url);

    if (invalid) {
      return res.status(400).json({
        ok: false,
        error: "Every item must include originalImageUrl, imageUrl, or mediaUrl.",
      });
    }

    const { error: itemsError } = await supabase
      .from("ai_job_items")
      .insert(jobItemsPayload);

    if (itemsError) throw itemsError;

    setImmediate(() => {
      processAiBulkJob(job.id).catch((error) => {
        console.error("Background AI bulk job failed:", error.message);
      });
    });

    res.json({
      ok: true,
      job: {
        id: job.id,
        status: "processing",
        totalItems: items.length,
        total_items: items.length,
      },
    });
  } catch (error) {
    res.status(500).json({
      ok: false,
      error: error.response?.data || error.message,
    });
  }
});

app.get("/api/ai/jobs", async (_req, res) => {
  try {
    requireSupabaseConfig();

    const { data, error } = await supabase
      .from("ai_jobs")
      .select("*")
      .order("created_at", { ascending: false });

    if (error) throw error;

    res.json({ ok: true, jobs: data || [] });
  } catch (error) {
    res.status(500).json({ ok: false, error: error.message });
  }
});

app.get("/api/ai/jobs/:id", async (req, res) => {
  try {
    requireSupabaseConfig();

    const { data: job, error: jobError } = await supabase
      .from("ai_jobs")
      .select("*")
      .eq("id", req.params.id)
      .single();

    if (jobError) throw jobError;

    const { data: items, error: itemsError } = await supabase
      .from("ai_job_items")
      .select("*")
      .eq("ai_job_id", req.params.id)
      .order("created_at", { ascending: true });

    if (itemsError) throw itemsError;

    res.json({ ok: true, job, items: items || [] });
  } catch (error) {
    res.status(500).json({ ok: false, error: error.message });
  }
});

app.get("/api/ai/results", async (_req, res) => {
  try {
    requireSupabaseConfig();

    const { data, error } = await supabase
      .from("media_assets")
      .select("*")
      .eq("is_ai_generated", true)
      .order("created_at", { ascending: false });

    if (error) throw error;

    res.json({ ok: true, results: (data || []).map(withAliases) });
  } catch (error) {
    res.status(500).json({ ok: false, error: error.message });
  }
});

app.post("/api/ai/save-result", async (req, res) => {
  try {
    requireSupabaseConfig();

    const editedImageUrl =
      req.body.editedImageUrl ||
      req.body.edited_image_url ||
      req.body.imageUrl ||
      req.body.image_url ||
      req.body.mediaUrl ||
      req.body.media_url;

    if (!editedImageUrl || !isPublicHttpsUrl(editedImageUrl)) {
      return res.status(400).json({
        ok: false,
        error: "editedImageUrl must be a public HTTPS URL.",
      });
    }

    const media = await insertAiGeneratedMedia({
      editedImageUrl,
      sourceMediaAssetId:
        req.body.sourceMediaAssetId || req.body.source_media_asset_id || null,
      aiJobId: req.body.aiJobId || req.body.ai_job_id || null,
      prompt: req.body.prompt || "",
      editMode: req.body.editMode || req.body.edit_mode || null,
      provider: req.body.provider || null,
      model: req.body.model || null,
      aspectRatio: req.body.aspectRatio || req.body.aspect_ratio || null,
      outputSize: req.body.outputSize || req.body.output_size || null,
      resolution: req.body.resolution || null,
      quality: req.body.quality || null,
    });

    res.json({ ok: true, media: withAliases(media) });
  } catch (error) {
    res.status(500).json({ ok: false, error: error.response?.data || error.message });
  }
});

app.delete("/api/ai/results/:id", async (req, res) => {
  try {
    requireSupabaseConfig();

    const { error } = await supabase
      .from("media_assets")
      .delete()
      .eq("id", req.params.id)
      .eq("is_ai_generated", true);

    if (error) throw error;

    res.json({ ok: true, deleted: true });
  } catch (error) {
    res.status(500).json({ ok: false, error: error.message });
  }
});

/**
 * INBOX / CHATS
 */
app.get("/api/inbox/conversations", async (req, res) => {
  try {
    requireSupabaseConfig();

    let query = supabase
      .from("ig_conversations")
      .select("*")
      .order("last_message_at", { ascending: false, nullsFirst: false });

    if (req.query.status) {
      query = query.eq("status", req.query.status);
    }

    if (req.query.search) {
      const search = `%${req.query.search}%`;
      query = query.or(`username.ilike.${search},last_message_text.ilike.${search}`);
    }

    const { data, error } = await query;

    if (error) throw error;

    res.json({ ok: true, conversations: data || [] });
  } catch (error) {
    res.status(500).json({ ok: false, error: error.message });
  }
});

app.post("/api/inbox/sync-conversations", async (_req, res) => {
  try {
    requireSupabaseConfig();
    requirePageMessagingConfig();

    const response = await axios.get(metaUrl(`/${FACEBOOK_PAGE_ID}/conversations`), {
      params: {
        platform: "instagram",
        fields: "id,updated_time,participants,messages.limit(1){id,message,from,to,created_time}",
        access_token: FACEBOOK_PAGE_ACCESS_TOKEN,
      },
      timeout: 30000,
    });

    const synced = [];

    for (const conv of response.data?.data || []) {
      const participant = conv.participants?.data?.find(
        (p) => String(p.id) !== String(FACEBOOK_PAGE_ID),
      );

      const lastMessage = conv.messages?.data?.[0];

      const { data, error } = await supabase
        .from("ig_conversations")
        .upsert(
          {
            platform: "instagram",
            meta_conversation_id: conv.id,
            ig_scoped_user_id: participant?.id || null,
            username: participant?.name || participant?.username || participant?.id || null,
            last_message_text: lastMessage?.message || "",
            last_message_at: lastMessage?.created_time || conv.updated_time || new Date().toISOString(),
            raw: conv,
          },
          { onConflict: "meta_conversation_id" },
        )
        .select()
        .single();

      if (!error) synced.push(data);
    }

    res.json({ ok: true, conversations: synced });
  } catch (error) {
    res.status(500).json({ ok: false, error: error.response?.data || error.message });
  }
});

app.get("/api/inbox/conversations/:id/messages", async (req, res) => {
  try {
    requireSupabaseConfig();

    const { data: conversation, error: convError } = await supabase
      .from("ig_conversations")
      .select("*")
      .eq("id", req.params.id)
      .single();

    if (convError) throw convError;

    if (req.query.sync === "true" && conversation.meta_conversation_id && FACEBOOK_PAGE_ACCESS_TOKEN) {
      const response = await axios.get(metaUrl(`/${conversation.meta_conversation_id}/messages`), {
        params: {
          fields: "id,message,from,to,created_time,attachments",
          access_token: FACEBOOK_PAGE_ACCESS_TOKEN,
        },
        timeout: 30000,
      });

      for (const msg of response.data?.data || []) {
        const fromId = msg.from?.id || null;
        const direction =
          String(fromId) === String(FACEBOOK_PAGE_ID) ? "outbound" : "inbound";
        const attachment = msg.attachments?.data?.[0] || null;

        await supabase.from("ig_messages").upsert(
          {
            conversation_id: conversation.id,
            meta_message_id: msg.id,
            meta_conversation_id: conversation.meta_conversation_id,
            ig_scoped_user_id: conversation.ig_scoped_user_id,
            direction,
            message_type: attachment ? attachment.type || "unknown" : "text",
            text: msg.message || "",
            media_url: attachment?.payload?.url || null,
            from_id: fromId,
            to_id: msg.to?.data?.[0]?.id || null,
            sent_at: msg.created_time || new Date().toISOString(),
            raw: msg,
          },
          { onConflict: "meta_message_id" },
        );
      }
    }

    const { data, error } = await supabase
      .from("ig_messages")
      .select("*")
      .eq("conversation_id", req.params.id)
      .order("sent_at", { ascending: true });

    if (error) throw error;

    res.json({ ok: true, conversation, messages: data || [] });
  } catch (error) {
    res.status(500).json({ ok: false, error: error.response?.data || error.message });
  }
});

app.post("/api/inbox/send-text", async (req, res) => {
  try {
    requireSupabaseConfig();
    requirePageMessagingConfig();

    const { conversationId, recipientId, text } = req.body;

    if (!recipientId || !text) {
      return res.status(400).json({
        ok: false,
        error: "recipientId and text are required.",
      });
    }

    const response = await axios.post(
      metaUrl(`/${FACEBOOK_PAGE_ID}/messages`),
      {
        recipient: { id: recipientId },
        messaging_type: "RESPONSE",
        message: { text },
      },
      {
        params: { access_token: FACEBOOK_PAGE_ACCESS_TOKEN },
        timeout: 30000,
      },
    );

    if (conversationId && isValidUuid(conversationId)) {
      await supabase.from("ig_messages").insert({
        conversation_id: conversationId,
        meta_message_id: response.data?.message_id || null,
        ig_scoped_user_id: recipientId,
        direction: "outbound",
        message_type: "text",
        text,
        from_id: FACEBOOK_PAGE_ID,
        to_id: recipientId,
        sent_at: new Date().toISOString(),
        raw: response.data,
      });

      await supabase
        .from("ig_conversations")
        .update({
          last_message_text: text,
          last_message_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        })
        .eq("id", conversationId);
    }

    res.json({ ok: true, result: response.data });
  } catch (error) {
    res.status(500).json({ ok: false, error: error.response?.data || error.message });
  }
});

app.post("/api/inbox/send-image", async (req, res) => {
  try {
    requireSupabaseConfig();
    requirePageMessagingConfig();

    const { conversationId, recipientId, imageUrl } = req.body;

    if (!recipientId || !imageUrl) {
      return res.status(400).json({
        ok: false,
        error: "recipientId and imageUrl are required.",
      });
    }

    const response = await axios.post(
      metaUrl(`/${FACEBOOK_PAGE_ID}/messages`),
      {
        recipient: { id: recipientId },
        messaging_type: "RESPONSE",
        message: {
          attachment: {
            type: "image",
            payload: {
              url: imageUrl,
              is_reusable: true,
            },
          },
        },
      },
      {
        params: { access_token: FACEBOOK_PAGE_ACCESS_TOKEN },
        timeout: 30000,
      },
    );

    if (conversationId && isValidUuid(conversationId)) {
      await supabase.from("ig_messages").insert({
        conversation_id: conversationId,
        meta_message_id: response.data?.message_id || null,
        ig_scoped_user_id: recipientId,
        direction: "outbound",
        message_type: "image",
        media_url: imageUrl,
        from_id: FACEBOOK_PAGE_ID,
        to_id: recipientId,
        sent_at: new Date().toISOString(),
        raw: response.data,
      });

      await supabase
        .from("ig_conversations")
        .update({
          last_message_text: "[image]",
          last_message_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        })
        .eq("id", conversationId);
    }

    res.json({ ok: true, result: response.data });
  } catch (error) {
    res.status(500).json({ ok: false, error: error.response?.data || error.message });
  }
});

/**
 * COMMENTS
 */
app.post("/api/comments/sync-media", async (_req, res) => {
  try {
    requireSupabaseConfig();

    const token = META_ACCESS_TOKEN || FACEBOOK_PAGE_ACCESS_TOKEN;

    if (!IG_USER_ID || !token) {
      return res.status(400).json({
        ok: false,
        error: "IG_USER_ID and access token are required.",
      });
    }

    const response = await axios.get(metaUrl(`/${IG_USER_ID}/media`), {
      params: {
        fields: "id,caption,media_type,media_url,permalink,thumbnail_url,timestamp,comments_count",
        access_token: token,
      },
      timeout: 30000,
    });

    const saved = [];

    for (const media of response.data?.data || []) {
      const { data, error } = await supabase
        .from("ig_media_cache")
        .upsert(
          {
            ig_media_id: media.id,
            caption: media.caption || null,
            media_type: media.media_type || null,
            media_url: media.media_url || null,
            permalink: media.permalink || null,
            thumbnail_url: media.thumbnail_url || null,
            timestamp: media.timestamp || null,
            comments_count: media.comments_count || 0,
            raw: media,
          },
          { onConflict: "ig_media_id" },
        )
        .select()
        .single();

      if (!error) saved.push(data);
    }

    res.json({ ok: true, media: saved });
  } catch (error) {
    res.status(500).json({ ok: false, error: error.response?.data || error.message });
  }
});

app.get("/api/comments/media", async (_req, res) => {
  try {
    requireSupabaseConfig();

    const { data, error } = await supabase
      .from("ig_media_cache")
      .select("*")
      .order("timestamp", { ascending: false });

    if (error) throw error;

    res.json({ ok: true, media: data || [] });
  } catch (error) {
    res.status(500).json({ ok: false, error: error.message });
  }
});

app.post("/api/comments/sync", async (req, res) => {
  try {
    requireSupabaseConfig();

    const igMediaId = req.body.igMediaId || req.body.ig_media_id;

    if (!igMediaId) {
      return res.status(400).json({ ok: false, error: "igMediaId is required." });
    }

    const token = META_ACCESS_TOKEN || FACEBOOK_PAGE_ACCESS_TOKEN;

    const response = await axios.get(metaUrl(`/${igMediaId}/comments`), {
      params: {
        fields: "id,text,username,timestamp,like_count,replies{id,text,username,timestamp,like_count}",
        access_token: token,
      },
      timeout: 30000,
    });

    const saved = [];

    for (const comment of response.data?.data || []) {
      const savedComment = await upsertCommentFromMeta({
        ...comment,
        media_id: igMediaId,
      });

      if (savedComment) saved.push(savedComment);

      for (const reply of comment.replies?.data || []) {
        const savedReply = await upsertCommentFromMeta({
          ...reply,
          media_id: igMediaId,
          parent_id: comment.id,
        });
        if (savedReply) saved.push(savedReply);
      }
    }

    res.json({ ok: true, comments: saved });
  } catch (error) {
    res.status(500).json({ ok: false, error: error.response?.data || error.message });
  }
});

app.get("/api/comments", async (req, res) => {
  try {
    requireSupabaseConfig();

    let query = supabase
      .from("ig_comments")
      .select("*")
      .order("timestamp", { ascending: false });

    if (req.query.igMediaId) {
      query = query.eq("ig_media_id", req.query.igMediaId);
    }

    if (req.query.unreplied === "true") {
      query = query.eq("replied_by_app", false).eq("private_replied_by_app", false);
    }

    if (req.query.search) {
      const search = `%${req.query.search}%`;
      query = query.or(`username.ilike.${search},text.ilike.${search}`);
    }

    if (req.query.limit) {
      query = query.limit(Number(req.query.limit));
    }

    const { data, error } = await query;

    if (error) throw error;

    res.json({ ok: true, comments: data || [] });
  } catch (error) {
    res.status(500).json({ ok: false, error: error.message });
  }
});

app.post("/api/comments/:commentId/reply", async (req, res) => {
  try {
    requireSupabaseConfig();

    const message = req.body.message;

    if (!message) {
      return res.status(400).json({ ok: false, error: "message is required." });
    }

    const result = await sendCommentPublicReply(req.params.commentId, message);

    await supabase
      .from("ig_comments")
      .update({
        replied_by_app: true,
        replied_at: new Date().toISOString(),
      })
      .eq("ig_comment_id", req.params.commentId);

    res.json({ ok: true, result });
  } catch (error) {
    res.status(500).json({ ok: false, error: error.response?.data || error.message });
  }
});

app.post("/api/comments/:commentId/private-reply", async (req, res) => {
  try {
    requireSupabaseConfig();

    const message = req.body.message;

    if (!message) {
      return res.status(400).json({ ok: false, error: "message is required." });
    }

    const result = await sendCommentPrivateReply(req.params.commentId, message);

    await supabase
      .from("ig_comments")
      .update({
        private_replied_by_app: true,
        private_replied_at: new Date().toISOString(),
      })
      .eq("ig_comment_id", req.params.commentId);

    res.json({ ok: true, result });
  } catch (error) {
    res.status(500).json({ ok: false, error: error.response?.data || error.message });
  }
});

app.post("/api/comments/:commentId/like", async (req, res) => {
  try {
    const token = META_ACCESS_TOKEN || FACEBOOK_PAGE_ACCESS_TOKEN;

    const response = await axios.post(metaUrl(`/${req.params.commentId}/likes`), null, {
      params: { access_token: token },
      timeout: 30000,
    });

    res.json({ ok: true, result: response.data });
  } catch (error) {
    res.status(400).json({
      ok: false,
      unsupported: true,
      error:
        error.response?.data ||
        "Instagram comment like is not supported by this API/token.",
    });
  }
});

app.post("/api/comments/:commentId/hide", async (req, res) => {
  try {
    const token = META_ACCESS_TOKEN || FACEBOOK_PAGE_ACCESS_TOKEN;
    const hide = req.body.hide !== false;

    const response = await axios.post(metaUrl(`/${req.params.commentId}`), null, {
      params: {
        hide,
        access_token: token,
      },
      timeout: 30000,
    });

    await supabase
      ?.from("ig_comments")
      .update({ is_hidden: hide })
      .eq("ig_comment_id", req.params.commentId);

    res.json({ ok: true, result: response.data });
  } catch (error) {
    res.status(500).json({ ok: false, error: error.response?.data || error.message });
  }
});

app.delete("/api/comments/:commentId", async (req, res) => {
  try {
    const token = META_ACCESS_TOKEN || FACEBOOK_PAGE_ACCESS_TOKEN;

    const response = await axios.delete(metaUrl(`/${req.params.commentId}`), {
      params: { access_token: token },
      timeout: 30000,
    });

    await supabase
      ?.from("ig_comments")
      .update({ is_deleted: true })
      .eq("ig_comment_id", req.params.commentId);

    res.json({ ok: true, result: response.data });
  } catch (error) {
    res.status(500).json({ ok: false, error: error.response?.data || error.message });
  }
});

/**
 * AUTO REPLY RULES
 */
app.get("/api/comments/auto-reply/rules", async (_req, res) => {
  try {
    requireSupabaseConfig();

    const { data, error } = await supabase
      .from("ig_auto_reply_rules")
      .select("*")
      .order("created_at", { ascending: false });

    if (error) throw error;

    res.json({ ok: true, rules: data || [] });
  } catch (error) {
    res.status(500).json({ ok: false, error: error.message });
  }
});

app.post("/api/comments/auto-reply/rules", async (req, res) => {
  try {
    requireSupabaseConfig();

    const body = req.body;

    const { data, error } = await supabase
      .from("ig_auto_reply_rules")
      .insert({
        name: body.name,
        is_enabled: body.isEnabled ?? body.is_enabled ?? false,
        trigger_type: body.triggerType || body.trigger_type || "any_comment",
        keywords: Array.isArray(body.keywords) ? body.keywords : [],
        reply_mode: body.replyMode || body.reply_mode || "private_reply",
        public_reply_text: body.publicReplyText || body.public_reply_text || null,
        private_reply_text: body.privateReplyText || body.private_reply_text || null,
        only_once_per_user:
          body.onlyOncePerUser ?? body.only_once_per_user ?? true,
      })
      .select()
      .single();

    if (error) throw error;

    res.json({ ok: true, rule: data });
  } catch (error) {
    res.status(500).json({ ok: false, error: error.message });
  }
});

app.patch("/api/comments/auto-reply/rules/:id", async (req, res) => {
  try {
    requireSupabaseConfig();

    const body = req.body;
    const update = {};

    if (body.name !== undefined) update.name = body.name;
    if (body.isEnabled !== undefined || body.is_enabled !== undefined) {
      update.is_enabled = body.isEnabled ?? body.is_enabled;
    }
    if (body.triggerType !== undefined || body.trigger_type !== undefined) {
      update.trigger_type = body.triggerType || body.trigger_type;
    }
    if (body.keywords !== undefined) update.keywords = body.keywords;
    if (body.replyMode !== undefined || body.reply_mode !== undefined) {
      update.reply_mode = body.replyMode || body.reply_mode;
    }
    if (body.publicReplyText !== undefined || body.public_reply_text !== undefined) {
      update.public_reply_text = body.publicReplyText ?? body.public_reply_text;
    }
    if (body.privateReplyText !== undefined || body.private_reply_text !== undefined) {
      update.private_reply_text = body.privateReplyText ?? body.private_reply_text;
    }
    if (body.onlyOncePerUser !== undefined || body.only_once_per_user !== undefined) {
      update.only_once_per_user = body.onlyOncePerUser ?? body.only_once_per_user;
    }

    const { data, error } = await supabase
      .from("ig_auto_reply_rules")
      .update(update)
      .eq("id", req.params.id)
      .select()
      .single();

    if (error) throw error;

    res.json({ ok: true, rule: data });
  } catch (error) {
    res.status(500).json({ ok: false, error: error.message });
  }
});

app.delete("/api/comments/auto-reply/rules/:id", async (req, res) => {
  try {
    requireSupabaseConfig();

    const { error } = await supabase
      .from("ig_auto_reply_rules")
      .delete()
      .eq("id", req.params.id);

    if (error) throw error;

    res.json({ ok: true, deleted: true });
  } catch (error) {
    res.status(500).json({ ok: false, error: error.message });
  }
});

/**
 * POSTS
 */
app.post("/api/posts", async (req, res) => {
  try {
    requireSupabaseConfig();

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
            details: error.details || null,
          });
        }
      }

      return res.json({
        ok: failed.length === 0,
        posts: created.map(withAliases),
        createdCount: created.length,
        failedCount: failed.length,
        failed,
      });
    }

    const post = await createScheduledPostFromInput(req.body);

    res.json({ ok: true, post: withAliases(post) });
  } catch (error) {
    res.status(error.statusCode || 500).json({
      ok: false,
      error: error.response?.data || error.message || String(error),
      details: error.details || null,
    });
  }
});

app.get("/api/posts", async (_req, res) => {
  try {
    requireSupabaseConfig();

    const { data, error } = await supabase
      .from("scheduled_posts")
      .select("*")
      .order("scheduled_at", { ascending: true });

    if (error) throw error;

    res.json({ ok: true, posts: (data || []).map(withAliases) });
  } catch (error) {
    res.status(500).json({ ok: false, error: error.message });
  }
});

app.patch("/api/posts/:id", async (req, res) => {
  try {
    requireSupabaseConfig();

    const update = {};

    if (req.body.caption !== undefined) update.caption = req.body.caption;
    if (req.body.hashtags !== undefined) update.hashtags = normalizeHashtags(req.body.hashtags);
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

    res.json({ ok: true, post: withAliases(data) });
  } catch (error) {
    res.status(500).json({ ok: false, error: error.message });
  }
});

app.delete("/api/posts/:id", async (req, res) => {
  try {
    requireSupabaseConfig();

    const { data, error } = await supabase
      .from("scheduled_posts")
      .update({
        status: "cancelled",
        updated_at: new Date().toISOString(),
      })
      .eq("id", req.params.id)
      .select()
      .single();

    if (error) throw error;

    res.json({ ok: true, post: withAliases(data) });
  } catch (error) {
    res.status(500).json({ ok: false, error: error.message });
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

    res.json({ ok: true, post: withAliases(updatedPost) });
  } catch (error) {
    res.status(500).json({ ok: false, error: error.response?.data || error.message });
  }
});

/**
 * CRON
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

    for (const post of duePosts || []) {
      try {
        await publishScheduledPost(post);
      } catch (error) {
        console.error(`Failed scheduled post ${post.id}:`, error.response?.data || error.message);
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
    error: `Endpoint not found: ${req.method} ${req.originalUrl}`,
  });
});

app.listen(PORT, () => {
  console.log(`AutoFlow Backend running on port ${PORT}`);
});
