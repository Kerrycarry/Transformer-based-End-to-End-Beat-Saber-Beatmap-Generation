import * as bsmap from '../../../BeatSaber-JSMap/src/mod.ts';
import type { IWrapInfo } from '../../../BeatSaber-JSMap/src/types/beatmap/wrapper/info.ts';
import type { IWrapBeatmap } from '../../../BeatSaber-JSMap/src/types/beatmap/wrapper/beatmap.ts';
import { join } from "https://deno.land/std@0.201.0/path/mod.ts";
import { exists } from "https://deno.land/std/fs/mod.ts";
import { basename } from "https://deno.land/std/path/mod.ts";
import { assert } from "https://deno.land/std/assert/mod.ts";


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

async function processBeatmap(meta: any, target_data: string[]) {
  //check if meta.status is in target_data
  if (!target_data.includes(meta.status)) {
    updateFailMeta(meta, meta.status);
    return;
  }
  bsmap.globals.directory = meta.beatmap_path || '';
  try {
    if (meta.audio_offset !== 0) {
      error[AUDIO_OFFSET].push(meta.id);
      updateFailMeta(meta, AUDIO_OFFSET);
      return;
    }

    const difficultyFile = bsmap.readDifficultyFileSync(meta.beatmap_dat_name);
    
    if (hasBpmFieldWithNonEmptyListRecursive(difficultyFile)) {
      error[BPM_EVENTS].push(meta.id);
      updateFailMeta(meta, BPM_EVENTS);
      return;
    }

    if (meta.editor_offset !== 0) {
      error[EDITOR_OFFSET].push(meta.id);
      updateFailMeta(meta, EDITOR_OFFSET);
      return;
    }

    const is_complex = await handleComplexBeats(meta, difficultyFile);
    if (is_complex) {
      error[COMPLEX_BEATS].push(meta.id);
      updateFailMeta(meta, COMPLEX_BEATS);
      return;
    }          
    updateProcessedMeta(meta, difficultyFile);

  } catch (error) {
    console.error("Error processing directory:", meta.beatmap_path,meta.beatmap_dat_name, error);
    error[DIR_ERROR].push(meta.id);
    updateFailMeta(meta, DIR_ERROR);
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
      difficulty.customData?._editorOffset ?? 0
    ]);
}

function getComplexBeats(difficultyFile: IWrapBeatmap): number[] {
  const beats = difficultyFile.colorNotes.map(note => note.time);
  return beats.filter(time => time % complexBeatNumber !== 0);
}

async function copyBeatmap(meta: any, difficultyFile: IWrapBeatmap) {
  // log the changes
  console.log("copy path:");
  const folderName = basename(meta.beatmap_path);
  const command = `robocopy "X:\\Beatmap\\${folderName}" "X:\\Beatmap2\\${folderName}" /E`;
  const command2 = `robocopy "X:\\Beatmap\\${folderName}\\processed" "X:\\Beatmap2\\${folderName}_processed" /E`;
  const command3 = `Remove-Item -Path "X:\\Beatmap2\\*" -Recurse -Force`
  console.log(command3);
  console.log(command); 
  console.log(command2);
  //copy the beatmap to processed folder
  const processedPath = `${meta.beatmap_path}/processed`;
  //copy info.dat and song to processed folder
  if (lastPath !== meta.beatmap_path) {
    const fileExists = await exists(processedPath); 
    if (!fileExists){await Deno.mkdir(processedPath);}
    await Deno.copyFile(`${meta.beatmap_path}/${meta.info_name}`, `${processedPath}/${meta.info_name}`);
    await Deno.copyFile(`${meta.beatmap_path}/${meta.song_name}`, `${processedPath}/${meta.song_name}`);
    //copy difficulty to processed folder
    const info = bsmap.readInfoFileSync(meta.info_name);
    const difficultyTuples = getDifficultyTuples(info);
    for (const difficultyTuple of difficultyTuples) {
      await Deno.copyFile(`${meta.beatmap_path}/${difficultyTuple[0]}`, `${processedPath}/${difficultyTuple[0]}`);
    }
  }
  // record lastPath to avoid redo
  lastPath = meta.beatmap_path
}

async function handleOffset(difficultyFile: IWrapBeatmap, meta: any){
  const offsets = difficultyFile.colorNotes.map(note => parseFloat((note.time % complexBeatNumber).toFixed(3)));
  //get offset stats
  const countMap = new Map<number, number>();
  for (const num of offsets) {
    countMap.set(num, (countMap.get(num) || 0) + 1);
  }
  // 找到出现次数最少的数
  let minCountNum: number[] = [];
  let minCount = Infinity;
  for (const [num, count] of countMap) {
    if (count < minCount) {
      minCount = count;
      minCountNum = [num]; // 重新赋值
    } else if (count === minCount) {
      minCountNum.push(num); // 追加
    }
  }
  // 找到出现最多的数
  let maxCountNum: number | null = null;
  let maxCount = 0;

  for (const [num, count] of countMap) {
    if (count > maxCount) {
      maxCount = count;
      maxCountNum = num;
    }
  }
  // 找到原列表中等于 minCountNum 的元素及其索引
  const indices : number[] = [];
  offsets.forEach((num, index) => {
    if (minCountNum.includes(num)) {
      indices.push(index);
    }
  });
  const minElementsWithIndex = indices.map(index => difficultyFile.colorNotes[index].time);
  console.log("***************offsets stats:");
  console.log(countMap);
  console.log("minElementsWithIndex:", minElementsWithIndex);
  console.log("maxCountNum, maxCount:", maxCountNum, maxCount);
  if(maxCountNum !== 0){
    console.log("maxCountNum!=0:", maxCountNum);
    console.log("maxCount/difficultyFile.colorNotes.length:", maxCount/difficultyFile.colorNotes.length);
  }
  if (countMap.size === 1 || (maxCountNum!==0 && maxCount > 0.7 * difficultyFile.colorNotes.length)) {
    //update offset to info.dat
    const processedPath = `${meta.beatmap_path}/processed`;
    bsmap.globals.directory = processedPath;
    const info = bsmap.readInfoFileSync(meta.info_name);
    info.difficulties.map(difficulty => {
      if (difficulty.characteristic === 'Standard' && difficulty.difficulty === meta.difficulty) {
        const offset = (maxCountNum as number) * 60 / meta.bpm * 1000;
        difficulty.customData._editorOffset = difficulty.customData._editorOffset + offset;
        difficulty.customData._editorOldOffset = difficulty.customData._editorOffset;
      }
    });
    await bsmap.writeInfoFile(info, 2, {
      directory: processedPath,
      filename: meta.info_name
    });
    // remove offset from difficulty.colorNotes
    difficultyFile.colorNotes.map(note => {note.time = note.time - (maxCountNum as number);});
  }
  return countMap.size
}

function handleSlide(difficultyFile: IWrapBeatmap){
  const tempSlideNote: Record<number, any[]> = {};
  const noteWithoutSlide = [];
  const noteWithSlide = [];
  // looking for slides
  for (const note of difficultyFile.colorNotes) {
    const currentColor = note.color;
    if (currentColor in tempSlideNote){
      const lastNote = tempSlideNote[currentColor].at(-1)
      // check if it is part of slide
      if (note.time > lastNote.time && note.time - lastNote.time < complexBeatNumber && (lastNote.direction == note.direction || note.direction == 8)){
        tempSlideNote[currentColor].push(note);
      }
      // identify it as a new note. Handle slide in temp
      else{
        noteWithoutSlide.push(tempSlideNote[currentColor][0]);
        if (tempSlideNote[currentColor].length > 1){
          noteWithSlide.push(tempSlideNote[currentColor].map(note => note.time));
        }
        tempSlideNote[currentColor] = [note];
      }
    }
    else{
      tempSlideNote[currentColor] = [note];
    }
  }
  // handle rest of note in tempSlideNote
  [0,1].map(currentColor => {
    noteWithoutSlide.push(tempSlideNote[currentColor][0]);
    if (tempSlideNote[currentColor].length > 1){
      noteWithSlide.push(tempSlideNote[currentColor].map(note => note.time));
    }
  });
  // update difficultyFile.colorNotes with note without slide
  difficultyFile.colorNotes = noteWithoutSlide
  console.log("noteWithSlide length:",noteWithSlide.length)
  console.log(noteWithSlide)
}
async function handleComplexBeats(meta: any, difficultyFile: IWrapBeatmap) {
  const complexBeats = getComplexBeats(difficultyFile);
  if (complexBeats.length === 0) {
    return false;
  }

  // handle floating point issue
  difficultyFile.colorNotes.map(note => {note.time = parseFloat(note.time.toFixed(3));});
  const complexBeatsRound = getComplexBeats(difficultyFile);
  if (complexBeatsRound.length === 0) {
    return false;
  }

  // copy beatmap file and edit it in the processed folder
  await copyBeatmap(meta, difficultyFile);

  // handle potential offset
  const countMapSize = await handleOffset(difficultyFile, meta);
  if(countMapSize === 1){
    return false;
  }

  // handle slide
  handleSlide(difficultyFile)
  const complexBeatsWithoutSlice = getComplexBeats(difficultyFile);
  console.log("complexBeatsWithoutSlice.length",complexBeatsWithoutSlice.length)
  console.log(complexBeatsWithoutSlice)
  if (complexBeatsWithoutSlice.length === 0) {
    return false;
  }
  

  return true;
}

function updateMeta(info: IWrapInfo, difficultyTuple: [string, string, number, number, number], dir: { name: string, fullPath: string }) {
  const id = dir.name.match(id_pattern)?.[0] || dir.name;
  
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
    editor_offset: difficultyTuple[4],
    audio_offset: info.audio.audioOffset,
    status: RAW_DATA,
  });
}

function updateFailMeta(meta: any, errorType: string) {
  meta.status = errorType;
  output_meta.push({
    ...meta
  });
}

function updateProcessedMeta(meta: any, difficultyFile: any) {
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
  meta.status = PROCESSED_DATA;
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
const DIR_ERROR: string = "Directory Error";
const AUDIO_OFFSET: string = "Audio Offset";
const EDITOR_OFFSET: string = "Editor Offset";
const BPM_EVENTS: string = "BPM Events"; // bpm, editor_offset, audio_offset, the level of these three issues is same
const FLOATING_ERROR: string = "Floating Error";
const MISSING_OFFSET: string = "Missing Offset";
const SMALL_COMPLEX: string = "1/64 note"
const COMPLEX_BEATS: string = "Complex Beats";//issues cannot be resolved are classfied as complex beats
const RAW_DATA: string = "Raw Data";
const PROCESSED_DATA: string = "Processed Data";

// create a dictionary to store the each error type
const error: Record<string, string[]> = {};
error[DIR_ERROR] = [];
error[AUDIO_OFFSET] = [];
error[EDITOR_OFFSET] = [];
error[BPM_EVENTS] = [];
error[FLOATING_ERROR] = [];
error[MISSING_OFFSET] = [];
error[SMALL_COMPLEX] = [];
error[COMPLEX_BEATS] = [];


const pathCount: Record<string, number> = {};
let load: number = 0;
const [directory, manifest_directory, pipeline, complex_beat_number] = Deno.args;
const complexBeatNumber = Number(complex_beat_number);
let lastPath: string | null = null;
const id_pattern = /^[a-zA-Z0-9]+/;

if(pipeline === "create_manifest"){
  const directories = await getDirectoriesWithPaths(directory || '');
  for (const dir of directories) {
    await processDirectory(dir);
  }

  const summary = {
    dir_num: directories.length,
    load: output_meta.length,
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
  const target_data: string[] = [COMPLEX_BEATS];
  const input_meta: JSON[] = await loadJsonl(manifest_directory)
  for (const meta of input_meta) {
    await processBeatmap(meta, target_data);
  }
  console.log("*****************************result:");
  console.log("Path count:", pathCount);


  // Summary
  const summary = {
    total_num: input_meta.length,
    load,
    error_num: {
      DIR_ERROR: error[DIR_ERROR].length,
      AUDIO_OFFSET: error[AUDIO_OFFSET].length,
      EDITOR_OFFSET: error[EDITOR_OFFSET].length,
      BPM_EVENTS: error[BPM_EVENTS].length,
      FLOATING_ERROR: error[FLOATING_ERROR].length,
      MISSING_OFFSET: error[MISSING_OFFSET].length,
      SMALL_COMPLEX: error[SMALL_COMPLEX].length,
      COMPLEX_BEATS: error[COMPLEX_BEATS].length,
    },
    error_type:{
      DIR_ERROR: error[DIR_ERROR],
      AUDIO_OFFSET: error[AUDIO_OFFSET],
      EDITOR_OFFSET: error[EDITOR_OFFSET],
      BPM_EVENTS: error[BPM_EVENTS],
      FLOATING_ERROR: error[FLOATING_ERROR],
      MISSING_OFFSET: error[MISSING_OFFSET],
      SMALL_COMPLEX: error[SMALL_COMPLEX],
      COMPLEX_BEATS: error[COMPLEX_BEATS],
    }
  };

  console.log(summary);
}

// Save metadata to file
const fileContent = output_meta.map((obj) => JSON.stringify(obj)).join("\n");
await Deno.writeTextFile(manifest_directory, fileContent);


