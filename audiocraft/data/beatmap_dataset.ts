import * as bsmap from '../../../BeatSaber-JSMap/src/mod.ts';
import type { IWrapInfo } from '../../../BeatSaber-JSMap/src/types/beatmap/wrapper/info.ts';
import type { IWrapBeatmap } from '../../../BeatSaber-JSMap/src/types/beatmap/wrapper/beatmap.ts';
import { join } from "https://deno.land/std@0.201.0/path/mod.ts";
import { exists } from "https://deno.land/std/fs/mod.ts";
import { basename } from "https://deno.land/std/path/mod.ts";
import { assert } from "https://deno.land/std/assert/mod.ts";
import { pLimit } from "https://deno.land/x/p_limit@v1.0.0/mod.ts";
import { copy } from 'https://deno.land/std@0.224.0/fs/copy.ts';


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
      // pathCount[currentPath] = (pathCount[currentPath] || 0) + 1;
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
const processBeatmap=(metaList: any[]) =>
  Promise.all(metaList.map((meta) => processSingleBeatmap(meta)))
  .then((processedMetaList: any[]) => {
    // handle inconsistent error type
    const foundError = [AUDIO_OFFSET, BPM_EVENTS, EDITOR_OFFSET].find(errorType => 
        processedMetaList.some(meta => meta.status === errorType)
    );
    if(foundError){
      processedMetaList = processedMetaList.map(item => ({
        ...item,
        status: foundError,
      }));
      for (const meta of processedMetaList){
        output_meta.push({...meta});
        result[foundError].push(meta.id);
      } 
      return;
    }
    // handle inconsistent offset
    const offsetList = processedMetaList.map(meta => meta?.process_used?.offset ?? 0);
    const firstOffset = offsetList[0]
    const meta = processedMetaList[0]
    // check if offset is same among different difficulties
    if(offsetList.some(offset => offset !== firstOffset) ){
      // only allow 0.012 difference measured in beat 
      const diff_threshold = 0.012;
      
      for (const offset of offsetList){
        const diff = Math.abs(firstOffset - offset)
        if(diff > diff_threshold){
          logPath(meta.beatmap_path, "")
          console.log(`first offset = ${firstOffset}, difference = ${diff}`)
          assert(false, `offset difference: > diff_threshold: ${diff_threshold}`)
        }
      }
    }
    // update offset to info.dat
    if(firstOffset !== 0){
      const processedPath = `${meta.beatmap_path}/processed`;
      const info = bsmap.readInfoFileSync(`${processedPath}/${meta.info_name}`);
      info.difficulties.map(difficulty => {
        if (difficulty.characteristic === 'Standard') {
          const offsetInSec = firstOffset * 60 / meta.bpm * 1000
          difficulty.customData._editorOffset = offsetInSec;
          difficulty.customData._editorOldOffset = offsetInSec;
        }
      });
      bsmap.writeInfoFile(info, 2, {
        directory: processedPath,
        filename: meta.info_name
      });
      // TODO: don't update offset to meta for now (change in the future version),
      // TODO：把更新offset从赋值改成+=
    }
    for (const meta of processedMetaList){
      output_meta.push({...meta});
      result[meta.status].push(meta.id);
    }
  });

function logPath(beatmapPath: string, difficulty: string){
  // log the changes
  console.log("*************************");
  console.log("Handling:");
  const folderName = basename(beatmapPath);
  const command1 = `X:\\Beatmap\\${folderName}`
  const command2 = `Remove-Item -Path "X:\\Beatmap2\\*" -Recurse -Force`
  const command3 = `robocopy "X:\\Beatmap\\${folderName}" "X:\\Beatmap2\\${folderName}" /E`;
  const command4 = `robocopy "X:\\Beatmap\\${folderName}\\processed" "X:\\Beatmap2\\${folderName}_processed" /E`;
  console.log(command1);
  console.log(difficulty)
  console.log(command2);
  console.log(command3); 
  console.log(command4);
}
function processSingleBeatmap(meta: any) {
  //check if meta.status is in target_data
  if (!target_data.includes(meta.status))
    return meta;
  logPath(meta.beatmap_path, meta.difficulty)
  try {
    if (meta.audio_offset !== 0) {
      meta.status = AUDIO_OFFSET;
      return meta;
    }

    const processedPath = `${meta.beatmap_path}/processed/${meta.beatmap_dat_name}`;
    const difficultyFile = bsmap.readDifficultyFileSync(processedPath);
    
    if (hasBpmFieldWithNonEmptyListRecursive(difficultyFile)) {
      meta.status = BPM_EVENTS;
      return meta;
    }

    if (meta.editor_offset !== 0) {
      meta.status = EDITOR_OFFSET;
      return meta;
    }

    const is_complex = handleComplexBeats(meta, difficultyFile);
    if (is_complex) {
      meta.status = COMPLEX_BEATS;
      return meta;
    }

    updateProcessedMeta(meta, difficultyFile);
    return meta;

  } catch (_error) {
    console.error(`Error processing beatmap ${meta.id}`, _error);
    const folderName = basename(meta.beatmap_path);
    console.log(`${folderName}\\${meta.difficulty}`)
    meta.status = UNKNOWN_ERROR;
    return meta
  }
}


const processDirectory = (dir: { name: string, fullPath: string }) =>
  getInfoPath(dir.fullPath)
  .then(infoPath => bsmap.readInfoFileSync(`${dir.fullPath}/${infoPath}`))
  .then(info => {
    const difficultyTuples = getDifficultyTuples(info);
    difficultyTuples.map(difficultyTuple => updateMeta(info, difficultyTuple, dir));
  })
  .catch (_error=>{
    result[UNKNOWN_ERROR].push(dir.name);
    logPath(dir.fullPath, "")
    console.error("Error processing directory:", dir.name, _error);
    }
  )

async function getInfoPath(dirPath: string): Promise<string> {
  const filePath = `${dirPath}/info.dat`;
  const fileExists = await exists(filePath); // Await the exists check
  return fileExists ? "info.dat" : "Info.dat";
}

function getDifficultyTuples(info: IWrapInfo): [string, string, number, number, number][] {
  return info.difficulties
    .filter(difficulty => difficulty.characteristic === 'Standard').reverse()
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

async function copyDifficulty(filePath: string, targetPath: string) {
  const targetString: string[] = ["_version","version"];    
  try {
      const data = await Deno.readTextFile(filePath);
      const json = JSON.parse(data);
      const keysList: string[] = Object.keys(json);
      // 判断特定字符串是否在列表中
      const allFound = targetString.every(target => keysList.includes(target));
      if(allFound){
          // 删除第一个字符为下划线的键
          for (const key of keysList) {
              if (key.startsWith('_')) {
              delete json[key];
              }
          }
      }
      
      await Deno.writeTextFile(targetPath, JSON.stringify(json, null, 2));
      
  } catch (_error) {
      console.error("Error reading or parsing JSON:", _error);
  }
}

async function copyBeatmap(metaList: any[]) {
  const meta = metaList[0]
  //copy the beatmap to processed folder
  const processedPath = `${meta.beatmap_path}/processed`;
  const fileExists = await exists(processedPath); 
  if (!fileExists)
    await Deno.mkdir(processedPath);
  //copy info.dat processed folder
  await Deno.copyFile(`${meta.beatmap_path}/${meta.info_name}`, `${processedPath}/${meta.info_name}`);
}

function handleOffset(difficultyFile: IWrapBeatmap, meta: any){
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
  console.log(`Offsets stats (${countMap.size} unique values):`);
  console.log(countMap);
  console.log("min Elements With Index:", minElementsWithIndex);
  console.log(`max Count Num: ${maxCountNum} (count:${maxCount})`);
  if(maxCountNum !== 0){
    console.log("max Count Num != 0:");
    console.log("max Count /difficultyFile.colorNotes.length:", maxCount/difficultyFile.colorNotes.length);
  }
  console.log()
  if (countMap.size === 1 || 
    (maxCountNum!==0 && 
      // meet strong condition or lose condition
      (maxCount/difficultyFile.colorNotes.length > 0.7 || (maxCount/difficultyFile.colorNotes.length > 0.6 && (difficultyFile.colorNotes[0].time - (maxCountNum as number))% complexBeatNumber === 0) ))) {
    //update offset to meta later in processBeatmap()
    meta.process_used.offset = (maxCountNum as number);
    // remove offset from difficulty.colorNotes
    difficultyFile.colorNotes.map(note => {note.time = note.time - (maxCountNum as number);});
  }
  return countMap
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
  const colorList: number[] = []
  if(0 in tempSlideNote){
    colorList.push(0)
  } 
  if(1 in tempSlideNote){
    colorList.push(1)
  }
  colorList.map(currentColor => {
    noteWithoutSlide.push(tempSlideNote[currentColor][0]);
    if (tempSlideNote[currentColor].length > 1){
      noteWithSlide.push(tempSlideNote[currentColor].map(note => note.time));
    }
  });
  // update difficultyFile.colorNotes with note without slide
  difficultyFile.colorNotes = noteWithoutSlide
  console.log(`Note with slide (length:",${noteWithSlide.length}):`)
  console.log(noteWithSlide)
  console.log()
}
function handleComplexBeats(meta: any, difficultyFile: IWrapBeatmap) {
  const complexBeats = getComplexBeats(difficultyFile);
  if (complexBeats.length === 0) {
    return false;
  }

  meta.process_used = {};
  // handle floating point issue
  difficultyFile.colorNotes.map(note => {note.time = parseFloat(note.time.toFixed(3));});
  const complexBeatsRound = getComplexBeats(difficultyFile);
  meta.process_used.round = complexBeats.length - complexBeatsRound.length
  if (complexBeatsRound.length === 0) {
    return false;
  }

  // handle slide
  handleSlide(difficultyFile)
  const complexBeatsWithoutSlice = getComplexBeats(difficultyFile);
  console.log(`complex Beats Without Slice (length: ${complexBeatsWithoutSlice.length})`)
  console.log(complexBeatsWithoutSlice)
  console.log()
  meta.process_used.slice = complexBeatsRound.length - complexBeatsWithoutSlice.length
  if (complexBeatsWithoutSlice.length === 0) {
    return false;
  }

  // handle potential offset
  const countMap= handleOffset(difficultyFile, meta);
  if(countMap.size === 1){
    return false;
  }

  // handle potential bpm event
  // if (countMap.size !== 1){
  //   // check if contain tuplet notes
  //   const keys = Array.from(countMap.keys());
  //   const hasOnly = keys.length === 3 && keys.includes(0) && keys.includes(0.042) && keys.includes(0.083);
  //   if(hasOnly)
  //     console.log("complex beats are all tuplet notes")
  //   else if(keys.includes(0.042) && countMap.get(0.042) > 2 && keys.includes(0.083) && countMap.get(0.083) > 2)
  //     console.log("complex beats has something beyond tuplet note")
  //   else
  //     console.log("complex beats has something else")
  // }
  
  const complexBeatsWithoutOffset = getComplexBeats(difficultyFile);
  console.log(`complex Beats Without offset (length: ${complexBeatsWithoutOffset.length})`)
  console.log(complexBeatsWithoutOffset)
  console.log()

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

const id_pattern = /^[a-zA-Z0-9]+/;
const [directory, manifest_directory, pipeline, complex_beat_number, target] = Deno.args;
const complexBeatNumber = Number(complex_beat_number);

//count bpm event
// const pathCount: Record<string, number> = {};

//all following variables are global updated
const output_meta: any[] = [];
const UNKNOWN_ERROR: string = "UNKNOWN_ERROR";
const AUDIO_OFFSET: string = "AUDIO_OFFSET";
const EDITOR_OFFSET: string = "EDITOR_OFFSET";
const BPM_EVENTS: string = "BPM_EVENTS"; // bpm, editor_offset, audio_offset, the level of these three issues is same
const FLOATING_ERROR: string = "Floating Error";
const MISSING_OFFSET: string = "MISSING_OFFSET";
const SMALL_COMPLEX: string = "SMALL_COMPLEX" // such as 1/64 note
const COMPLEX_BEATS: string = "COMPLEX_BEATS";//issues cannot be resolved are classfied as complex beats
const RAW_DATA: string = "RAW_DATA";
const PROCESSED_DATA: string = "PROCESSED_DATA";

// create a dictionary to store the each error type
const result: Record<string, string[]> = {};
result[UNKNOWN_ERROR] = [];
result[AUDIO_OFFSET] = [];
result[EDITOR_OFFSET] = [];
result[BPM_EVENTS] = [];
result[FLOATING_ERROR] = [];
result[MISSING_OFFSET] = [];
result[SMALL_COMPLEX] = [];
result[COMPLEX_BEATS] = [];
result[PROCESSED_DATA] = [];

const target_data: string[] = [target];

const start = performance.now();
if(pipeline === "create_manifest"){
  const directories = await getDirectoriesWithPaths(directory || '');
  await Promise.all(directories.map(dir => processDirectory(dir)))
  const summary = {
    dir_num: directories.length,
    load: output_meta.length,
    error_num: {
      UNKNOWN_ERROR: result[UNKNOWN_ERROR].length,
    },
    error_type:{
      UNKNOWN_ERROR: result[UNKNOWN_ERROR],
    }
  };
  console.log(summary);
}
else{
  const input_meta: JSON[] = await loadJsonl(manifest_directory)
  const input_meta_group: Record<string, any[]> = {};
  for (const meta of input_meta) {
    const currentPath: string = (meta as any).beatmap_path;
    (input_meta_group[currentPath] ||= []).push(meta);
  }
  // copy beatmap files to processed folder, and save all beatmap changes there
  await Promise.all(Object.entries(input_meta_group).map(([key, value])=>copyBeatmap(value)));
  // copying song and difficulty need much longer time
  const limit = pLimit(5);
  // await Promise.all(Object.entries(input_meta_group).map(([key, value])=>limit(()=>Deno.copyFile(`${value[0].beatmap_path}/${value[0].song_name}`, `${value[0].beatmap_path}/processed/${value[0].song_name}`))));
  // await Promise.all(input_meta.map((meta: any)=> limit(() => copyDifficulty(`${meta.beatmap_path}/${meta.beatmap_dat_name}`, `${meta.beatmap_path}/processed/${meta.beatmap_dat_name}`))));
  await Promise.all(Object.entries(input_meta_group).map(([key, value])=>processBeatmap(value)));
  
  // console.log("*****************************result:");
  // console.log("Path count:", pathCount);

  // Summary
  const summary = {
    total_num: input_meta.length,
    error_num: {
      PROCESSED_DATA: result[PROCESSED_DATA].length,
      UNKNOWN_ERROR: result[UNKNOWN_ERROR].length,
      AUDIO_OFFSET: result[AUDIO_OFFSET].length,
      EDITOR_OFFSET: result[EDITOR_OFFSET].length,
      BPM_EVENTS: result[BPM_EVENTS].length,
      FLOATING_ERROR: result[FLOATING_ERROR].length,
      MISSING_OFFSET: result[MISSING_OFFSET].length,
      SMALL_COMPLEX: result[SMALL_COMPLEX].length,
      COMPLEX_BEATS: result[COMPLEX_BEATS].length,
    },
    error_type:{
      PROCESSED_DATA: result[PROCESSED_DATA],
      UNKNOWN_ERROR: result[UNKNOWN_ERROR],
      AUDIO_OFFSET: result[AUDIO_OFFSET],
      EDITOR_OFFSET: result[EDITOR_OFFSET],
      BPM_EVENTS: result[BPM_EVENTS],
      FLOATING_ERROR: result[FLOATING_ERROR],
      MISSING_OFFSET: result[MISSING_OFFSET],
      SMALL_COMPLEX: result[SMALL_COMPLEX],
      COMPLEX_BEATS: result[COMPLEX_BEATS],
    }
  };

  console.log(summary);
}

const end = performance.now();
console.log(`代码执行时间: ${((end - start)/1000).toFixed(2)} 秒`);

// Save metadata to file
const fileContent = output_meta.map((obj) => JSON.stringify(obj)).join("\n");
await Deno.writeTextFile(manifest_directory, fileContent);


