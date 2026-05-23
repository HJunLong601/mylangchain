import { randomUUID } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import type { CreateLearningNoteInput, LearningNote, LearningNoteKind } from "../types.js";

// 这里先用 JSON 文件做本地持久化。
//
// 为什么不一开始就上数据库？
// 1. 当前阶段重点是 Agent 主链路，不希望数据库配置干扰学习。
// 2. JSON 文件足够观察“工具调用 -> 保存笔记 -> 再检索笔记”的闭环。
// 3. 后续切到 SQLite / PostgreSQL 时，只需要替换这个 store 层，上层 Agent 不必大改。
const currentDir = dirname(fileURLToPath(import.meta.url));
const notesFilePath = join(currentDir, "..", "data", "learning-notes.json");

const DEFAULT_NOTE_KIND: LearningNoteKind = "observation";

async function ensureStoreFile() {
  // 确保 data 目录存在。这样首次运行时不会因为没有目录而写入失败。
  await mkdir(dirname(notesFilePath), { recursive: true });

  try {
    await readFile(notesFilePath, "utf-8");
  } catch {
    // 如果文件不存在，就初始化为空数组。
    await writeFile(notesFilePath, "[]", "utf-8");
  }
}

async function readNotesFromFile(): Promise<LearningNote[]> {
  await ensureStoreFile();
  const raw = await readFile(notesFilePath, "utf-8");

  try {
    const data = JSON.parse(raw) as LearningNote[];
    return Array.isArray(data) ? data : [];
  } catch {
    // 文件被手动改坏时，不让整个 Agent 崩掉。
    // 生产环境这里应该记录错误日志，并提示管理员修复数据。
    return [];
  }
}

async function writeNotesToFile(notes: LearningNote[]) {
  await ensureStoreFile();
  await writeFile(notesFilePath, JSON.stringify(notes, null, 2), "utf-8");
}

function normalizeTags(tags: string[] | undefined) {
  // 前端可能传入 [" langgraph ", "", "agent"]，这里统一清洗一下。
  return [...new Set((tags ?? []).map((tag) => tag.trim()).filter(Boolean))];
}

export async function listLearningNotes(): Promise<LearningNote[]> {
  const notes = await readNotesFromFile();

  // 新笔记排在前面，符合日常使用习惯。
  return notes.sort(
    (a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime(),
  );
}

export async function createLearningNote(
  input: CreateLearningNoteInput,
): Promise<LearningNote> {
  const now = new Date().toISOString();
  const note: LearningNote = {
    id: `note_${randomUUID()}`,
    title: input.title.trim() || "未命名学习笔记",
    content: input.content.trim(),
    kind: input.kind ?? DEFAULT_NOTE_KIND,
    tags: normalizeTags(input.tags),
    createdAt: now,
    updatedAt: now,
  };

  const notes = await listLearningNotes();
  await writeNotesToFile([note, ...notes]);
  return note;
}

export async function searchLearningNotes(query: string, limit = 5): Promise<LearningNote[]> {
  const keywords = query
    .toLowerCase()
    .split(/\s+|，|,|。|\?|？/)
    .map((keyword) => keyword.trim())
    .filter(Boolean);

  if (keywords.length === 0) {
    return [];
  }

  const notes = await listLearningNotes();

  return notes
    .map((note) => {
      const searchableText = [
        note.title,
        note.content,
        note.kind,
        note.tags.join(" "),
      ].join(" ").toLowerCase();

      // 这里先用最简单的关键词命中计分。
      // 后续接 RAG 时，这个位置会替换成 embedding 检索 / 向量数据库查询。
      const score = keywords.reduce(
        (total, keyword) => total + (searchableText.includes(keyword) ? 1 : 0),
        0,
      );

      return {
        note,
        score,
      };
    })
    .filter((item) => item.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, limit)
    .map((item) => item.note);
}
