import * as bsmap from '../../../BeatSaber-JSMap/src/mod.ts';
import type { IWrapInfo } from '../../../BeatSaber-JSMap/src/types/beatmap/wrapper/info.ts';
import type { IWrapBeatmap } from '../../../BeatSaber-JSMap/src/types/beatmap/wrapper/beatmap.ts';
import { join } from "https://deno.land/std@0.201.0/path/mod.ts";
import { exists } from "https://deno.land/std/fs/mod.ts";

function hasBpmFieldWithNonEmptyListRecursive(
  instance: any,
  path: string = ""
): boolean {
  if (typeof instance !== "object" || instance === null) {
    return false; // 如果不是对象或为空，直接返回
  }

  let found = false;

  for (const key of Object.keys(instance)) {
    const value = instance[key];
    const lowerKey = key.toLowerCase(); // 转为小写
    const currentPath = path ? `${path}.${key}` : key; // 拼接当前字段的路径

    // 检查当前字段名和值
    if (
      lowerKey.includes("bpm") && // 不区分大小写检查
      Array.isArray(value) &&
      value.length > 0
    ) {
      // 更新统计字典
      pathCount[currentPath] = (pathCount[currentPath] || 0) + 1;

      console.log(`Found matching field: ${currentPath} with value:`);
      found = true; // 标记找到
    }

    // 如果当前字段是一个对象，递归检查它的字段
    if (typeof value === "object" && value !== null) {
      found = hasBpmFieldWithNonEmptyListRecursive(value, currentPath) || found;
    }
  }

  return found;
}

function getDecimalPlaces(num: number): number {
  const numStr = num.toString();
  return numStr.includes('.') ? numStr.split('.')[1].length : 0;
}

async function getDirectoriesWithPaths(dirPath: string) {
  const entries = [];
  for await (const entry of Deno.readDir(dirPath)) {
    if (entry.isDirectory) {
      const fullPath = join(dirPath, entry.name);
      entries.push({ name: entry.name, fullPath });
    }
  }
  return entries;
}

function processBeatmap(meta: any) {
  bsmap.globals.directory = meta.beatmap_path || '';
  try {
    const info = bsmap.readInfoFileSync(meta.info_name);
    
    if (info.audio.audioOffset !== 0) {
      error[AUDIO_OFFSET].push(meta.id);
      return;
    }

    const difficultyFile = bsmap.readDifficultyFileSync(meta.beatmap_dat_name);
    
    if (hasBpmFieldWithNonEmptyListRecursive(difficultyFile)) {
      error[BPM_EVENTS].push(meta.id);
      return;
    }

    if (meta.editor_offset !== 0) {
      error[EDITOR_OFFSET].push(meta.id);
      return;
    }

    const complexBeats = getComplexBeats(difficultyFile);
    if (complexBeats.length > 0) {
      handleComplexBeats(complexBeats, meta.id);
      return;
    }

    updateNewMeta(meta, difficultyFile);

  } catch (error) {
    console.error("Error processing directory:", meta.beatmap_path, error);
    error[DIR_ERROR].push(meta.id);
  }
}


async function processDirectory(dir: { name: string, fullPath: string }) {
  bsmap.globals.directory = dir.fullPath || '';
  try {
    const infoPath = await getInfoPath(dir.fullPath);
    const info = bsmap.readInfoFileSync(infoPath);

    const difficultyTuples = getDifficultyTuples(info);

    for (const difficultyTuple of difficultyTuples) {

      updateMeta(info, difficultyTuple, dir);
    }
    
  } catch (error) {
    console.error("Error processing directory:", dir.name, error);
    error[DIR_ERROR].push(dir.name);
  }
}

async function getInfoPath(dirPath: string): Promise<string> {
  const filePath = `${dirPath}/info.dat`;
  const fileExists = await exists(filePath); // Await the exists check
  return fileExists ? "info.dat" : "Info.dat";
}

function getDifficultyTuples(info: IWrapInfo): [string, string, number, number, number][] {
  return info.difficulties
    .filter(difficulty => difficulty.characteristic === 'Standard')
    .map(difficulty => [
      difficulty.filename,
      difficulty.difficulty,
      difficulty.njs,
      difficulty.njsOffset,
      difficulty.customData._editorOffset
    ]);
}

function getComplexBeats(difficultyFile: IWrapBeatmap): number[] {
  const beats = difficultyFile.colorNotes.map(note => note.time);
  return beats.filter(time => time % complexBeatNumber !== 0);
}

function handleComplexBeats(complexBeats: number[], id: string ) {
  const timeDecimalPlaces = complexBeats.map(time => getDecimalPlaces(time));
  if (timeDecimalPlaces.some(num => num > 6)) {
    error[FLOATING_ERROR].push(id);
    return;
  }

  const offsets = complexBeats.map(time => time % complexBeatNumber);
  if (offsets.every((val, _, arr) => val === arr[0])) {
    error[MISSING_OFFSET].push(id);
    return;
  }

  if (complexBeats.length / complexBeatNumber > 0.2) {
    console.log("Complex beat detected:", `${id}`, complexBeats);
    complexCount++;
  }

  error[COMPLEX_BEATS].push(id);
}

function updateMeta(info: IWrapInfo, difficultyTuple: [string, string, number, number, number], dir: { name: string, fullPath: string }) {
  const id = dir.name.match(/^[a-zA-Z0-9]+/)?.[0] || dir.name;
  
  output_meta.push({
    id: `${id}_${difficultyTuple[1]}`,
    beatmap_path: dir.fullPath,
    info_name: "Info.dat",
    song_name: info.audio.filename,
    bpm: info.audio.bpm,
    beatmap_dat_name: difficultyTuple[0],
    difficulty: difficultyTuple[1],
    njs: difficultyTuple[2],
    njsoffset: difficultyTuple[3],
    editor_offset: difficultyTuple[4]
  });
  load++;
}

function updateNewMeta(meta: any, difficultyFile: any) {
  const jsonData = JSON.stringify(difficultyFile);
  const difficultyPath = meta.beatmap_path+"/"+meta.difficulty+".json"
  Deno.writeTextFile(difficultyPath, jsonData);
  meta.beatmap_json_name = meta.difficulty+".json";
  meta.note_num = {
    colorNotes: difficultyFile.colorNotes.length,
    bombNotes: difficultyFile.bombNotes.length,
    obstacles: difficultyFile.obstacles.length,
    arcs: difficultyFile.arcs.length,
    chains: difficultyFile.chains.length,
  };
  output_meta.push({
    ...meta
  });
  load++;
}
const loadJsonl = async (filePath: string) => {
  // 读取 JSONL 文件的内容
  const data = await Deno.readTextFile(filePath);
  
  // 按行拆分，并将每行解析为 JSON 对象
  const jsonObjects = data.split('\n').map(line => {
    try {
      return JSON.parse(line);
    } catch (error) {
      console.error('Invalid JSON on line:', line);
      return null;
    }
  }).filter(Boolean); // 过滤掉解析失败的行
  
  return jsonObjects;
};

const output_meta: any[] = [];
let load: number = 0;
const DIR_ERROR: string = "Directory Error";
const AUDIO_OFFSET: string = "Audio Offset";
const EDITOR_OFFSET: string = "Editor Offset";
const BPM_EVENTS: string = "BPM Events";
const MISSING_OFFSET: string = "Missing Offset";
const FLOATING_ERROR: string = "Floating Error";
const COMPLEX_BEATS: string = "Complex Beats";

// create a dictionary to store the each error type
const error: Record<string, string[]> = {};
error[DIR_ERROR] = [];
error[AUDIO_OFFSET] = [];
error[EDITOR_OFFSET] = [];
error[BPM_EVENTS] = [];
error[MISSING_OFFSET] = [];
error[FLOATING_ERROR] = [];
error[COMPLEX_BEATS] = [];

const pathCount: Record<string, number> = {};
let complexCount: number = 0;

const [directory, manifest_directory, pipeline, complex_beat_number] = Deno.args;
const complexBeatNumber = Number(complex_beat_number);

if(pipeline === "create_manifest"){
  const directories = await getDirectoriesWithPaths(directory || '');
  for (const dir of directories) {
    await processDirectory(dir);
  }

  const summary = {
    dir_num: directories.length,
    load,
    error_num: {
      DIR_ERROR: error[DIR_ERROR].length,
    },
    error_type:{
      DIR_ERROR: error[DIR_ERROR],
    }
  };
  console.log(summary);
}
else{
  const input_meta: JSON[] = await loadJsonl(manifest_directory)
  input_meta.forEach(meta => {
    processBeatmap(meta)
  });
  console.log("*****************************result:");
  console.log("Path count:", pathCount);
  console.log("Noncomplex count:", complexCount);

  // Summary
  const summary = {
    total_num: input_meta.length,
    load,
    error_num: {
      DIR_ERROR: error[DIR_ERROR].length,
      AUDIO_OFFSET: error[AUDIO_OFFSET].length,
      EDITOR_OFFSET: error[EDITOR_OFFSET].length,
      BPM_EVENTS: error[BPM_EVENTS].length,
      MISSING_OFFSET: error[MISSING_OFFSET].length,
      FLOATING_ERROR: error[FLOATING_ERROR].length,
      COMPLEX_BEATS: error[COMPLEX_BEATS].length,
    },
    error_type:{
      DIR_ERROR: error[DIR_ERROR],
      AUDIO_OFFSET: error[AUDIO_OFFSET],
      EDITOR_OFFSET: error[EDITOR_OFFSET],
      BPM_EVENTS: error[BPM_EVENTS],
      MISSING_OFFSET: error[MISSING_OFFSET],
      FLOATING_ERROR: error[FLOATING_ERROR],
      COMPLEX_BEATS: error[COMPLEX_BEATS],
    }
  };

  console.log(summary);
}

// Save metadata to file
const fileContent = output_meta.map((obj) => JSON.stringify(obj)).join("\n");
await Deno.writeTextFile(manifest_directory, fileContent);


