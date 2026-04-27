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

const INSTAGRAM_GRAPH_HOST =
  process.env.GRAPH_HOST || "https://graph.instagram.com";
const META_GRAPH_HOST =
  process.env.META_GRAPH_HOST || "https://graph.facebook.com";

const GRAPH_VERSION =
  process.env.GRAPH_VERSION || process.env.META_GRAPH_VERSION || "v25.0";

const META_ACCESS_TOKEN = process.env.META_ACCESS_TOKEN;
const IG_USER_ID = process.env.IG_USER_ID;

const META_APP_SECRET = process.env.META_APP_SECRET;
const META_WEBHOOK_VERIFY_TOKEN = process.env.META_WEBHOOK_VERIFY_TOKEN;

const FACEBOOK_PAGE_ID = process.env.FACEBOOK_PAGE_ID;
const FACEBOOK_PAGE_ACCESS_TOKEN = process.env.FACEBOOK_PAGE_ACCESS_TOKEN;

const AUTO_REPLY_ENABLED =
  String(process.env.AUTO_REPLY_ENABLED || "false").toLowerCase() === "true";

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const GEMINI_TEXT_MODEL = process.env.GEMINI_TEXT_MODEL || "gemini-2.5-flash";
const GEMINI_IMAGE_MODEL =
  process.env.GEMINI_IMAGE_MODEL || "gemini-2.5-flash-image-preview";

const AI_DEFAULT_PROVIDER = process.env.AI_DEFAULT_PROVIDER || "gemini";
const AI_BULK_CONCURRENCY = Number(process.env.AI_BULK_CONCURRENCY || 2);

const DEFAULT_BUSINESS_NAME =
  process.env.DEFAULT_BUSINESS_NAME || "Flower Center";
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

function requireInstagramConfig() {
  if (!IG_USER_ID || !META_ACCESS_TOKEN) {
    throw new Error("Missing IG_USER_ID or META_ACCESS_TOKEN.");
  }
}

function requireGeminiConfig() {
  if (!GEMINI_API_KEY) {
    throw new Error("Missing GEMINI_API_KEY.");
  }
}

function cleanAccessToken(token) {
  return String(token || "")
    .trim()
    .replace(/^Bearer\s+/i, "")
    .replace(/^["']|["']$/g, "")
    .replace(/\s+/g, "");
}

function getInstagramToken() {
  return cleanAccessToken(META_ACCESS_TOKEN);
}

function getPageToken() {
  return cleanAccessToken(FACEBOOK_PAGE_ACCESS_TOKEN);
}

function isLikelyInstagramToken(token) {
  const clean = cleanAccessToken(token);
  return clean.startsWith("IG") || clean.startsWith("IGA");
}

function isLikelyFacebookToken(token) {
  const clean = cleanAccessToken(token);
  return clean.startsWith("EA");
}

function hasInstagramTokenConfig() {
  return Boolean(IG_USER_ID && getInstagramToken());
}

function hasPageTokenConfig() {
  return Boolean(FACEBOOK_PAGE_ID && getPageToken());
}

function buildGraphUrl(host, pathValue) {
  const cleanPath = String(pathValue).startsWith("/")
    ? String(pathValue)
    : `/${pathValue}`;

  return `${host}/${GRAPH_VERSION}${cleanPath}`;
}

function instagramGraphUrl(pathValue) {
  return buildGraphUrl(INSTAGRAM_GRAPH_HOST, pathValue);
}

function metaGraphUrl(pathValue) {
  return buildGraphUrl(META_GRAPH_HOST, pathValue);
}

function createGraphError(pathValue, errors) {
  const error = new Error(`Graph request failed for ${pathValue}.`);
  error.details = {
    path: pathValue,
    errors,
  };
  return error;
}

/**
 * CRITICAL FIX:
 * - If token starts IG / IGA => use graph.instagram.com only.
 * - If token starts EA => use graph.facebook.com first.
 * This prevents "Cannot parse access token" when IG token is sent to Meta Graph.
 */
async function graphGet(pathValue, params = {}, options = {}) {
  const token = cleanAccessToken(
    params.access_token ||
      options.accessToken ||
      META_ACCESS_TOKEN ||
      FACEBOOK_PAGE_ACCESS_TOKEN,
  );

  if (!token) {
    throw new Error("Missing access token.");
  }

  const mergedParams = {
    ...params,
    access_token: token,
  };

  const errors = [];
  const tokenIsInstagram = isLikelyInstagramToken(token);
  const tokenIsFacebook = isLikelyFacebookToken(token);

  let hostOrder = [];

  if (tokenIsInstagram) {
    hostOrder = ["instagram"];
  } else if (tokenIsFacebook) {
    hostOrder = options.preferInstagram ? ["instagram", "meta"] : ["meta", "instagram"];
  } else {
    hostOrder = options.preferInstagram ? ["instagram", "meta"] : ["meta", "instagram"];
  }

  for (const hostType of hostOrder) {
    const url =
      hostType === "instagram"
        ? instagramGraphUrl(pathValue)
        : metaGraphUrl(pathValue);

    try {
      return await axios.get(url, {
        params: mergedParams,
        timeout: options.timeout || 30000,
      });
    } catch (error) {
      const err = error.response?.data || error.message;

      errors.push({
        host: hostType,
        url,
        error: err,
      });

      console.warn(
        `${hostType === "instagram" ? "Instagram" : "Meta"} Graph GET failed ${pathValue}:`,
        err,
      );

      if (tokenIsInstagram && hostType === "instagram") break;
    }
  }

  throw createGraphError(pathValue, errors);
}

async function graphPost(pathValue, data = null, params = {}, options = {}) {
  const token = cleanAccessToken(
    params.access_token ||
      options.accessToken ||
      META_ACCESS_TOKEN ||
      FACEBOOK_PAGE_ACCESS_TOKEN,
  );

  if (!token) {
    throw new Error("Missing access token.");
  }

  const mergedParams = {
    ...params,
    access_token: token,
  };

  const errors = [];
  const tokenIsInstagram = isLikelyInstagramToken(token);
  const tokenIsFacebook = isLikelyFacebookToken(token);

  let hostOrder = [];

  if (tokenIsInstagram) {
    hostOrder = ["instagram"];
  } else if (tokenIsFacebook) {
    hostOrder = options.preferInstagram ? ["instagram", "meta"] : ["meta", "instagram"];
  } else {
    hostOrder = options.preferInstagram ? ["instagram", "meta"] : ["meta", "instagram"];
  }

  for (const hostType of hostOrder) {
    const url =
      hostType === "instagram"
        ? instagramGraphUrl(pathValue)
        : metaGraphUrl(pathValue);

    try {
      return await axios.post(url, data, {
        params: mergedParams,
        timeout: options.timeout || 30000,
      });
    } catch (error) {
      const err = error.response?.data || error.message;

      errors.push({
        host: hostType,
        url,
        error: err,
      });

      console.warn(
        `${hostType === "instagram" ? "Instagram" : "Meta"} Graph POST failed ${pathValue}:`,
        err,
      );

      if (tokenIsInstagram && hostType === "instagram") break;
    }
  }

  throw createGraphError(pathValue, errors);
}

function cleanupFile(filePath) {
  try {
    if (filePath && fs.existsSync(filePath)) {
      fs.unlinkSync(filePath);
    }
  } catch {
    // ignore cleanup errors
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

  if (clean.endsWith(".mp4") || clean.endsWith(".mov") || clean.endsWith(".m4v")) {
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

  if (["gemini", "openai", "openrouter"].includes(provider)) return provider;

  return "gemini";
}

function getProviderApiKey({ provider, apiKey }) {
  const cleanKey = apiKey ? String(apiKey).trim() : "";

  if (cleanKey.length > 0) return cleanKey;

  if (provider === "gemini") return GEMINI_API_KEY;

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
    conversationId: row.conversation_id ?? row.conversationId,
    igScopedUserId: row.ig_scoped_user_id ?? row.igScopedUserId,
    metaConversationId: row.meta_conversation_id ?? row.metaConversationId,
    lastMessageText: row.last_message_text ?? row.lastMessageText,
    lastMessageAt: row.last_message_at ?? row.lastMessageAt,
    unreadCount: row.unread_count ?? row.unreadCount,
    messageType: row.message_type ?? row.messageType,
    sentAt: row.sent_at ?? row.sentAt,
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

    if (error) console.warn("Could not fetch media asset by UUID:", error.message);
    if (data) return data;
  }

  if (mediaUrl) {
    const { data, error } = await supabase
      .from("media_assets")
      .select("*")
      .eq("media_url", mediaUrl)
      .maybeSingle();

    if (error) console.warn("Could not fetch media asset by media_url:", error.message);
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
    .jpeg({ quality: 92, mozjpeg: true })
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

async function pollMediaContainerStatus(containerId) {
  await sleep(3000);

  for (let attempt = 1; attempt <= 24; attempt += 1) {
    try {
      const response = await axios.get(instagramGraphUrl(`/${containerId}`), {
        params: {
          fields: "status_code,status",
          access_token: getInstagramToken(),
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
  requireInstagramConfig();

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
    access_token: getInstagramToken(),
  };

  if (finalMediaType === "image") {
    params.image_url = finalMediaUrl;
    params.media_type = "IMAGE";
  }

  if (finalMediaType === "video") {
    params.video_url = finalMediaUrl;
    params.media_type = "VIDEO";
  }

  const containerResponse = await axios.post(
    instagramGraphUrl(`/${IG_USER_ID}/media`),
    null,
    {
      params,
      timeout: 60000,
    },
  );

  const creationId = containerResponse.data?.id;

  if (!creationId) throw new Error("Meta did not return creation_id.");

  await pollMediaContainerStatus(creationId);

  let publishResponse = null;

  for (let attempt = 1; attempt <= 3; attempt += 1) {
    try {
      publishResponse = await axios.post(
        instagramGraphUrl(`/${IG_USER_ID}/media_publish`),
        null,
        {
          params: {
            creation_id: creationId,
            access_token: getInstagramToken(),
          },
          timeout: 60000,
        },
      );

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

  if (!publishResponse?.data?.id) throw new Error("Meta did not return publish ID.");

  return {
    creationId,
    publishId: publishResponse.data.id,
    mediaType: finalMediaType,
    mediaUrl: finalMediaUrl,
  };
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
  const safeHashtagCount = Number.isFinite(Number(hashtagCount))
    ? Math.max(3, Math.min(25, Number(hashtagCount)))
    : DEFAULT_HASHTAG_COUNT;

  const preset = captionPreset || "Luxury Product Caption";
  const custom = customPrompt ? `Custom instruction:\n${customPrompt}` : "";

  return `
You are a senior Instagram marketing copywriter for ${businessName}, a ${location}-based company specializing in premium artificial trees, artificial flowers, custom greenery, and luxury decor installations.

Analyze the image carefully and write content that matches the actual visible product.

Caption preset:
${preset}

${custom}

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
  "hashtags": ["#tag1", "#tag2"],
  "alt_text": "short alt text",
  "detected_product": "what the image shows",
  "visual_description": "short visual analysis",
  "marketing_angle": "main sales angle"
}

Rules:
- Caption must be specific to the image.
- Do not use placeholders.
- Do not invent discounts, offers, guarantees, or prices.
- Do not mention AI.
- Arabic must sound natural and premium.
- Use no more than ${safeHashtagCount} hashtags.
`;
}

async function generateVisionWithGemini({
  imageUrl,
  prompt,
  modelName = GEMINI_TEXT_MODEL,
  fallbackTextOnly = false,
}) {
  requireGeminiConfig();

  const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
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
  imageUrl,
  language = "arabic",
  tone = "premium",
  model = GEMINI_TEXT_MODEL,
  captionPreset = "Luxury Product Caption",
  customPrompt = "",
  businessName = DEFAULT_BUSINESS_NAME,
  location = DEFAULT_LOCATION,
  cta = DEFAULT_CTA,
  hashtagCount = DEFAULT_HASHTAG_COUNT,
  fallbackTextOnly = false,
}) {
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

  const text = await generateVisionWithGemini({
    imageUrl,
    prompt,
    modelName: model || GEMINI_TEXT_MODEL,
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
  const rules = [
    "The result must be photorealistic and look like real professional photography.",
    "Keep natural perspective, believable scale, correct shadows, realistic lighting, and harmonious colors.",
    "No cartoon, no painting, no CGI look, no artificial AI artifacts.",
    "No text, no watermark.",
  ];

  if (preserveProduct) {
    rules.push(
      "Keep the original tree / plant / flower arrangement exactly unchanged.",
      "Do NOT change product shape, trunk, branches, leaves, flowers, colors, density, height, width, proportions, angle, or realism.",
    );
  }

  if (keepPot) {
    rules.push(
      "Keep the pot / planter / base exactly unchanged unless explicitly asked.",
      "Do NOT change pot shape, color, material, texture, size, or placement.",
    );
  }

  rules.push(
    `Output aspect ratio: ${aspectRatio}.`,
    `Target output size: ${outputSize}.`,
    `Target resolution: ${resolution}.`,
    `Quality level: ${quality}.`,
  );

  return `
Professional AI image edit request.

Preset:
${preset}

Edit mode:
${editMode}

Background intensity:
${backgroundIntensity}

User instruction:
${userPrompt || customPrompt || "Create a premium realistic background that enhances the product."}

Fixed rules:
${rules.map((rule) => `- ${rule}`).join("\n")}

Return only the edited image.
`;
}

async function editImageWithGemini({ originalImageUrl, prompt, model = GEMINI_IMAGE_MODEL }) {
  requireGeminiConfig();

  if (!originalImageUrl || !isPublicHttpsUrl(originalImageUrl)) {
    throw new Error("originalImageUrl must be a public HTTPS URL.");
  }

  if (!prompt) throw new Error("prompt is required.");

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
    throw new Error(textOutput || "Gemini image model did not return image data.");
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
    await supabase
      .from("scheduled_posts")
      .update({
        status: "failed",
        error_message: JSON.stringify(error.response?.data || error.details || error.message),
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

function verifyMetaSignature(req) {
  if (!META_APP_SECRET) return true;

  const signature = req.headers["x-hub-signature-256"];

  if (!signature || !signature.startsWith("sha256=")) return false;

  const expected = `sha256=${createHmac("sha256", META_APP_SECRET)
    .update(req.rawBody || Buffer.from(""))
    .digest("hex")}`;

  const sigBuffer = Buffer.from(signature);
  const expectedBuffer = Buffer.from(expected);

  if (sigBuffer.length !== expectedBuffer.length) return false;

  return timingSafeEqual(sigBuffer, expectedBuffer);
}

async function storeWebhookEvent({
  body,
  field = null,
  eventType = null,
  processed = false,
  errorMessage = null,
}) {
  if (!supabase) return null;

  const { data, error } = await supabase
    .from("meta_webhook_events")
    .insert({
      object_type: body?.object || null,
      field,
      event_type: eventType,
      meta_id: null,
      payload: body,
      processed,
      error_message: errorMessage,
    })
    .select()
    .single();

  if (error) {
    console.warn("Webhook event insert failed:", error.message);
    return null;
  }

  return data;
}

async function fetchCommentDetails(commentId) {
  const response = await graphGet(
    `/${commentId}`,
    {
      fields: "id,text,username,timestamp,like_count,parent_id,media,from",
      access_token: getInstagramToken() || getPageToken(),
    },
    { preferInstagram: true },
  );

  return response.data;
}

async function upsertCommentFromMeta(value) {
  requireSupabaseConfig();

  let commentValue = value || {};
  const igCommentId = commentValue.id || commentValue.comment_id;

  if (!igCommentId) return null;

  let text = commentValue.text || commentValue.message || "";

  if (!text) {
    try {
      const details = await fetchCommentDetails(igCommentId);
      commentValue = { ...details, ...commentValue };
      text = commentValue.text || commentValue.message || "";
    } catch (error) {
      console.warn("Could not fetch comment details:", error.details || error.message);
    }
  }

  const mediaObject = commentValue.media || {};
  const fromObject = commentValue.from || commentValue.user || {};

  const row = {
    ig_comment_id: igCommentId,
    ig_media_id:
      commentValue.media_id ||
      mediaObject.id ||
      commentValue.ig_media_id ||
      null,
    parent_comment_id:
      commentValue.parent_id ||
      commentValue.parent_comment_id ||
      commentValue.parent?.id ||
      null,
    username:
      fromObject.username ||
      commentValue.username ||
      commentValue.user?.username ||
      null,
    user_id:
      fromObject.id ||
      commentValue.user_id ||
      commentValue.user?.id ||
      null,
    text,
    like_count: Number(commentValue.like_count || 0),
    timestamp: commentValue.timestamp
      ? new Date(commentValue.timestamp).toISOString()
      : new Date().toISOString(),
    is_reply: Boolean(commentValue.parent_id || commentValue.parent_comment_id),
    raw: commentValue,
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

async function saveMediaCache(media) {
  requireSupabaseConfig();

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

  if (error) throw error;

  return data;
}

async function syncCommentsForMedia(igMediaId) {
  requireSupabaseConfig();

  const token = getInstagramToken() || getPageToken();

  const response = await graphGet(
    `/${igMediaId}/comments`,
    {
      fields:
        "id,text,username,timestamp,like_count,replies{id,text,username,timestamp,like_count}",
      access_token: token,
    },
    { preferInstagram: true },
  );

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

  return saved;
}

async function upsertConversationFromMessage({ senderId, text, sentAt }) {
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
  });

  if (!conversation) return;

  const payload = {
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
  };

  if (mid) {
    const { error } = await supabase
      .from("ig_messages")
      .upsert(payload, { onConflict: "meta_message_id" });

    if (error) console.warn("Inbound message upsert failed:", error.message);
  } else {
    const { error } = await supabase.from("ig_messages").insert(payload);

    if (error) console.warn("Inbound message insert failed:", error.message);
  }
}

function extractParticipant(conversation) {
  const candidates =
    conversation.participants?.data ||
    conversation.participants ||
    conversation.users?.data ||
    [];

  if (!Array.isArray(candidates)) return null;

  return (
    candidates.find((p) => {
      const id = String(p.id || "");
      return id !== String(IG_USER_ID) && id !== String(FACEBOOK_PAGE_ID);
    }) ||
    candidates[0] ||
    null
  );
}

function normalizeMessageDirection(msg) {
  const fromId = String(msg.from?.id || msg.from_id || "");

  if (fromId === String(IG_USER_ID) || fromId === String(FACEBOOK_PAGE_ID)) {
    return "outbound";
  }

  return "inbound";
}

function normalizeMessageAttachment(msg) {
  const attachments = msg.attachments?.data || msg.attachments || [];

  if (Array.isArray(attachments) && attachments.length > 0) {
    return attachments[0];
  }

  return null;
}

async function upsertConversationRow(conversation) {
  requireSupabaseConfig();

  const participant = extractParticipant(conversation);
  const lastMessage =
    conversation.messages?.data?.[0] || conversation.messages?.[0] || null;

  const metaConversationId = conversation.id;
  const igScopedUserId = participant?.id || conversation.ig_scoped_user_id || null;

  const username =
    participant?.username ||
    participant?.name ||
    conversation.username ||
    igScopedUserId ||
    null;

  const { data, error } = await supabase
    .from("ig_conversations")
    .upsert(
      {
        platform: "instagram",
        meta_conversation_id: metaConversationId,
        ig_scoped_user_id: igScopedUserId,
        username,
        last_message_text: lastMessage?.message || lastMessage?.text || "",
        last_message_at:
          lastMessage?.created_time ||
          conversation.updated_time ||
          conversation.last_message_at ||
          new Date().toISOString(),
        status: "open",
      },
      { onConflict: "meta_conversation_id" },
    )
    .select()
    .single();

  if (error) throw error;

  return data;
}

async function saveConversationMessages(conversationRow, messages) {
  requireSupabaseConfig();

  let count = 0;

  for (const msg of messages || []) {
    const attachment = normalizeMessageAttachment(msg);
    const messageType = attachment?.type || (msg.message || msg.text ? "text" : "unknown");
    const mediaUrl = attachment?.payload?.url || attachment?.url || null;
    const direction = normalizeMessageDirection(msg);
    const metaMessageId = msg.id || msg.mid || null;

    const payload = {
      conversation_id: conversationRow.id,
      meta_message_id: metaMessageId,
      meta_conversation_id: conversationRow.meta_conversation_id,
      ig_scoped_user_id: conversationRow.ig_scoped_user_id,
      direction,
      message_type: messageType,
      text: msg.message || msg.text || "",
      media_url: mediaUrl,
      from_id: msg.from?.id || null,
      to_id: msg.to?.data?.[0]?.id || msg.to?.id || null,
      sent_at: msg.created_time || msg.sent_at || new Date().toISOString(),
      raw: msg,
    };

    let error = null;

    if (metaMessageId) {
      const result = await supabase
        .from("ig_messages")
        .upsert(payload, { onConflict: "meta_message_id" });

      error = result.error;
    } else {
      const result = await supabase.from("ig_messages").insert(payload);
      error = result.error;
    }

    if (error) {
      console.warn("Message save failed:", error.message);
    } else {
      count += 1;
    }
  }

  return count;
}

async function fetchMessagesForConversation(conversationId) {
  const token = getInstagramToken() || getPageToken();

  try {
    const response = await graphGet(
      `/${conversationId}/messages`,
      {
        fields: "id,message,from,to,created_time,attachments",
        access_token: token,
      },
      { preferInstagram: true },
    );

    return response.data?.data || [];
  } catch (firstError) {
    console.warn(
      "Conversation messages direct fetch failed, trying expanded conversation:",
      firstError.details || firstError.response?.data || firstError.message,
    );

    const response = await graphGet(
      `/${conversationId}`,
      {
        fields:
          "messages.limit(50){id,message,from,to,created_time,attachments},participants,updated_time",
        access_token: token,
      },
      { preferInstagram: true },
    );

    return response.data?.messages?.data || [];
  }
}

async function syncInstagramConversations() {
  requireSupabaseConfig();
  requireInstagramConfig();

  const errors = [];
  let conversationsCount = 0;
  let messagesCount = 0;

  const response = await graphGet(
    `/${IG_USER_ID}/conversations`,
    {
      fields:
        "id,participants,updated_time,messages.limit(1){id,message,from,to,created_time,attachments}",
      access_token: getInstagramToken(),
    },
    { preferInstagram: true },
  );

  const conversations = response.data?.data || [];

  for (const conversation of conversations) {
    try {
      const conversationRow = await upsertConversationRow(conversation);
      conversationsCount += 1;

      try {
        const messages = await fetchMessagesForConversation(conversation.id);
        messagesCount += await saveConversationMessages(conversationRow, messages);
      } catch (messageError) {
        errors.push({
          conversationId: conversation.id,
          stage: "messages",
          error: messageError.details || messageError.response?.data || messageError.message,
        });
      }
    } catch (conversationError) {
      errors.push({
        conversationId: conversation.id,
        stage: "conversation",
        error:
          conversationError.details ||
          conversationError.response?.data ||
          conversationError.message,
      });
    }
  }

  return {
    conversationsCount,
    messagesCount,
    errors,
  };
}

async function syncPageConversations() {
  requireSupabaseConfig();

  if (!hasPageTokenConfig()) {
    throw new Error("FACEBOOK_PAGE_ID and FACEBOOK_PAGE_ACCESS_TOKEN are required.");
  }

  const errors = [];
  let conversationsCount = 0;
  let messagesCount = 0;

  const response = await axios.get(metaGraphUrl(`/${FACEBOOK_PAGE_ID}/conversations`), {
    params: {
      platform: "instagram",
      fields:
        "id,updated_time,participants,messages.limit(1){id,message,from,to,created_time,attachments}",
      access_token: getPageToken(),
    },
    timeout: 30000,
  });

  const conversations = response.data?.data || [];

  for (const conversation of conversations) {
    try {
      const conversationRow = await upsertConversationRow(conversation);
      conversationsCount += 1;

      try {
        const messagesResponse = await axios.get(
          metaGraphUrl(`/${conversation.id}/messages`),
          {
            params: {
              fields: "id,message,from,to,created_time,attachments",
              access_token: getPageToken(),
            },
            timeout: 30000,
          },
        );

        messagesCount += await saveConversationMessages(
          conversationRow,
          messagesResponse.data?.data || [],
        );
      } catch (messageError) {
        errors.push({
          conversationId: conversation.id,
          stage: "messages",
          error: messageError.response?.data || messageError.message,
        });
      }
    } catch (conversationError) {
      errors.push({
        conversationId: conversation.id,
        stage: "conversation",
        error: conversationError.response?.data || conversationError.message,
      });
    }
  }

  return {
    conversationsCount,
    messagesCount,
    errors,
  };
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
  const response = await graphPost(
    `/${igCommentId}/replies`,
    null,
    {
      message,
      access_token: getInstagramToken() || getPageToken(),
    },
    { preferInstagram: true },
  );

  return response.data;
}

async function sendCommentPrivateReply(igCommentId, message) {
  const response = await graphPost(
    `/${igCommentId}/private_replies`,
    null,
    {
      message,
      access_token: getInstagramToken() || getPageToken(),
    },
    { preferInstagram: true },
  );

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
      if (
        (rule.reply_mode === "public_reply" || rule.reply_mode === "both") &&
        rule.public_reply_text
      ) {
        await sendCommentPublicReply(commentRow.ig_comment_id, rule.public_reply_text);
      }

      if (
        (rule.reply_mode === "private_reply" || rule.reply_mode === "both") &&
        rule.private_reply_text
      ) {
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

/**
 * ROOT / HEALTH
 */
app.get("/", (_req, res) => {
  res.json({
    ok: true,
    app: "AutoFlow Backend",
    status: "running",
    graphVersion: GRAPH_VERSION,
    instagramTokenConfigured: hasInstagramTokenConfig(),
    instagramTokenPrefix: getInstagramToken()
      ? `${getInstagramToken().slice(0, 4)}...`
      : null,
    igUserIdConfigured: Boolean(IG_USER_ID),
    pageTokenConfigured: hasPageTokenConfig(),
    pageTokenPrefix: getPageToken() ? `${getPageToken().slice(0, 4)}...` : null,
    supabaseConfigured: Boolean(supabase),
    cloudinaryConfigured: Boolean(
      CLOUDINARY_CLOUD_NAME && CLOUDINARY_API_KEY && CLOUDINARY_API_SECRET,
    ),
    aiStudio: true,
    instagramLoginMode: hasInstagramTokenConfig(),
    inboxEnabled: hasInstagramTokenConfig() || hasPageTokenConfig(),
    pageMessagingEnabled: hasPageTokenConfig(),
    commentsEnabled: Boolean((getInstagramToken() || getPageToken()) && IG_USER_ID),
    webhooksEnabled: Boolean(META_WEBHOOK_VERIFY_TOKEN),
  });
});

app.get("/api/health", (_req, res) => {
  res.json({
    ok: true,
    timestamp: new Date().toISOString(),
    graphVersion: GRAPH_VERSION,
    instagramTokenConfigured: hasInstagramTokenConfig(),
    instagramTokenPrefix: getInstagramToken()
      ? `${getInstagramToken().slice(0, 4)}...`
      : null,
    igUserIdConfigured: Boolean(IG_USER_ID),
    pageTokenConfigured: hasPageTokenConfig(),
    pageTokenPrefix: getPageToken() ? `${getPageToken().slice(0, 4)}...` : null,
    supabaseConfigured: Boolean(supabase),
    cloudinaryConfigured: Boolean(
      CLOUDINARY_CLOUD_NAME && CLOUDINARY_API_KEY && CLOUDINARY_API_SECRET,
    ),
    aiStudio: true,
    instagramLoginMode: hasInstagramTokenConfig(),
    inboxEnabled: hasInstagramTokenConfig() || hasPageTokenConfig(),
    pageMessagingEnabled: hasPageTokenConfig(),
    commentsEnabled: Boolean((getInstagramToken() || getPageToken()) && IG_USER_ID),
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

    const event = await storeWebhookEvent({
      body,
      field: null,
      eventType: "webhook",
      processed: false,
    });

    for (const entry of body.entry || []) {
      for (const messagingEvent of entry.messaging || []) {
        await insertInboundMessageFromWebhook(messagingEvent);
      }

      for (const change of entry.changes || []) {
        await storeWebhookEvent({
          body: change,
          field: change.field,
          eventType: "change",
          processed: false,
        });

        const field = String(change.field || "").toLowerCase();

        if (field.includes("comment")) {
          await upsertCommentFromMeta(change.value || {});
        }

        if (field.includes("message")) {
          await storeWebhookEvent({
            body: change,
            field: change.field,
            eventType: "message_change",
            processed: true,
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
    console.error("Webhook processing failed:", error.response?.data || error.details || error.message);

    await storeWebhookEvent({
      body: req.body,
      field: null,
      eventType: "webhook_error",
      processed: false,
      errorMessage: error.response?.data
        ? JSON.stringify(error.response.data)
        : JSON.stringify(error.details || error.message),
    });

    return res.status(500).json({
      ok: false,
      error: error.response?.data || error.details || error.message,
    });
  }
});

/**
 * DEBUG
 */
app.get("/api/debug/token-status", (_req, res) => {
  const instagramToken = getInstagramToken();
  const pageToken = getPageToken();

  res.json({
    ok: true,
    instagramTokenConfigured: Boolean(instagramToken),
    instagramTokenPrefix: instagramToken ? `${instagramToken.slice(0, 6)}...` : null,
    instagramTokenLooksLikeInstagram: isLikelyInstagramToken(instagramToken),
    instagramTokenLooksLikeFacebook: isLikelyFacebookToken(instagramToken),
    pageTokenConfigured: Boolean(pageToken),
    pageTokenPrefix: pageToken ? `${pageToken.slice(0, 6)}...` : null,
    pageTokenLooksLikeFacebook: isLikelyFacebookToken(pageToken),
  });
});

app.get("/api/debug/webhook-events", async (_req, res) => {
  try {
    requireSupabaseConfig();

    const { data, error } = await supabase
      .from("meta_webhook_events")
      .select("*")
      .order("created_at", { ascending: false })
      .limit(50);

    if (error) throw error;

    res.json({ ok: true, events: data || [] });
  } catch (error) {
    res.status(500).json({ ok: false, error: error.message });
  }
});

app.get("/api/debug/comments", async (_req, res) => {
  try {
    requireSupabaseConfig();

    const { count, error: countError } = await supabase
      .from("ig_comments")
      .select("*", { count: "exact", head: true });

    if (countError) throw countError;

    const { data, error } = await supabase
      .from("ig_comments")
      .select("*")
      .order("timestamp", { ascending: false })
      .limit(20);

    if (error) throw error;

    res.json({ ok: true, count, comments: data || [] });
  } catch (error) {
    res.status(500).json({ ok: false, error: error.message });
  }
});

app.get("/api/debug/inbox", async (_req, res) => {
  try {
    requireSupabaseConfig();

    const { count: conversationsCount, error: convCountError } = await supabase
      .from("ig_conversations")
      .select("*", { count: "exact", head: true });

    if (convCountError) throw convCountError;

    const { count: messagesCount, error: msgCountError } = await supabase
      .from("ig_messages")
      .select("*", { count: "exact", head: true });

    if (msgCountError) throw msgCountError;

    const { data, error } = await supabase
      .from("ig_conversations")
      .select("*")
      .order("last_message_at", { ascending: false, nullsFirst: false })
      .limit(10);

    if (error) throw error;

    res.json({
      ok: true,
      conversationsCount,
      messagesCount,
      conversations: data || [],
    });
  } catch (error) {
    res.status(500).json({ ok: false, error: error.message });
  }
});

/**
 * META TEST
 */
app.get("/api/meta/test-connection", async (_req, res) => {
  try {
    requireInstagramConfig();

    const response = await graphGet(
      "/me",
      {
        fields: "user_id,username",
        access_token: getInstagramToken(),
      },
      { preferInstagram: true },
    );

    res.json({
      ok: true,
      account: response.data,
      configuredIgUserId: IG_USER_ID,
      idMatches: String(response.data.user_id) === String(IG_USER_ID),
      graphVersion: GRAPH_VERSION,
    });
  } catch (error) {
    res.status(500).json({
      ok: false,
      error: error.details || error.response?.data || error.message,
    });
  }
});

app.post("/api/meta/test-connection", async (req, res) => {
  req.url = "/api/meta/test-connection";
  app.handle(req, res);
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

    const uploaded =
      mediaType === "image" ? await uploadNormalizedImage(file) : await uploadVideo(file);

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
 * PUBLISH
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
      error: error.details || error.response?.data || error.message,
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
 * AI IMAGE EDIT
 */
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
    const resolution = Number(req.body.resol
