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
function filterBeatmap(meta: any){
  if (meta.audio_offset !== 0) {
    meta.status = AUDIO_OFFSET;
    return;
  }
  const processedPath = `${meta.beatmap_path}/processed/${meta.beatmap_name}`;
  const difficultyFile = bsmap.readDifficultyFileSync(processedPath);
  if (hasBpmFieldWithNonEmptyListRecursive(difficultyFile)) {
    meta.status = BPM_EVENTS;
    return meta;
  }
  return difficultyFile;
}
function updateComplexMeta(difficultyFileList: IWrapBeatmap[], metaList: any[], process: string, log: boolean = false){
  const complexBeatsList = difficultyFileList.map(difficultyFile => getComplexBeats(difficultyFile))
  //log complex beat
  if(log)
    complexBeatsList.map(complexBeat=>{
      console.log(`complex Beats after process ${process} (length: ${complexBeat.length})`);
      console.log(`length!=0: ${complexBeat.length != 0}, length<10: ${complexBeat.length < 10}`);
      console.log(complexBeat);
    });
  //update meta in accordance with complexBeatsList length
  metaList.map((item, index) => {
    if (complexBeatsList[index].length < processedThreshold)
      item.status = PROCESSED_DATA;
    else
      item.status = COMPLEX_BEATS;
  });
  //early return condition
  if(complexBeatsList.every(complexBeats => complexBeats.length === 0))
    return true
  return false
}

function handleComplexBeats(difficultyFileList: IWrapBeatmap[], metaList: any[]){
  if(updateComplexMeta(difficultyFileList, metaList, "initial"))
    return;
  //handle slide
  difficultyFileList.map(difficultyFile => handleSlide(difficultyFile))
  if(updateComplexMeta(difficultyFileList, metaList, "handleSlide"))
    return;
  //handle potential offset
  // const offset = handleOffset(difficultyFileList, metaList[0])
  // log stats
  // metaList.forEach(item => {
  //   item.additional_offset = offset;
  // });
  // updateComplexMeta(difficultyFileList, metaList, "handleOffset", true);
}
function handleEditorOffset(difficultyFileList: IWrapBeatmap[], metaList: any[]){
  const editorOffsetList = metaList.map(meta => meta.editor_offset)
  const foundOffset = editorOffsetList.find(offset => offset !== 0);
  if(foundOffset){
    //get unique element in the editorOffsetList
    const uniqueEditorOffsetList = [...new Set(editorOffsetList)];
    if (uniqueEditorOffsetList.length > 1)  
      return false;
    const editorOffset = uniqueEditorOffsetList.find(offset => offset !== 0);
    const editorOffsetBeat = editorOffset / 1000 / 60 * metaList[0].bpm;
    //update editor offset to difficultyFileList
    difficultyFileList.map(difficultyFile => difficultyFile.colorNotes.map(note => {note.time = note.time - editorOffsetBeat;}));
    //log editor offset stat
    console.log(`editorOffsetList: ${editorOffsetList}`);
    console.log(`uniqueEditorOffsetList: ${uniqueEditorOffsetList} unique length: ${uniqueEditorOffsetList.length}`);
    console.log(`editor offset chosen: ${editorOffset}`);
  }
  return true;
}
function updateMeta2(metaList: any[], status: string = "", difficultyFileList: any[] = []){
  if(status !== ""){
    metaList = metaList.map(item => ({
      ...item,
      status: status,
    }));
  }
  if(difficultyFileList.length !== 0){
    assert(status === "", "status should be empty");
    metaList.map((meta, index) => {
      if (meta.status === PROCESSED_DATA){
        const difficultyFile = difficultyFileList[index];
        meta.processed_beatmap_json = meta.difficulty+".json";
        meta.note_num = {
          colorNotes: difficultyFile.colorNotes.length,
          bombNotes: difficultyFile.bombNotes.length,
          obstacles: difficultyFile.obstacles.length,
          arcs: difficultyFile.arcs.length,
          chains: difficultyFile.chains.length,
        };
        const jsonData = JSON.stringify(difficultyFile);
        const difficultyPath = meta.beatmap_path+"/processed/"+meta.difficulty+".json"
        difficulty_cache.push([difficultyPath, jsonData])
        // Deno.writeTextFile(difficultyPath, jsonData);
      }
    });
  }
  for (const meta of metaList){
    output_meta.push({...meta});
    result[meta.status].push(meta.id);
  } 
}
function processBeatmap(metaList: any[]){
  logPath(metaList[0].beatmap_path, "")
  try {
    //filter unwanted status
    const statusList = metaList.map(meta => meta.status);
    if (statusList.some(status => !target_data.includes(status))){
      updateMeta2(metaList); 
      return;
    }
    //filter AUDIO_OFFSET, BPM_EVENTS
    const difficultyFileList = metaList.map((meta) => filterBeatmap(meta))
    const foundError = [AUDIO_OFFSET, BPM_EVENTS].find(errorType => 
      metaList.some(meta => meta.status === errorType)
    );
    if(foundError){
      updateMeta2(metaList, foundError);
      return;
    }
    const isOffsetUnique = handleEditorOffset(difficultyFileList, metaList);
    if(!isOffsetUnique){
      updateMeta2(metaList, EDITOR_OFFSET);
      return;
    }
    handleComplexBeats(difficultyFileList, metaList);
    updateMeta2(metaList, "", difficultyFileList);
  }
  catch (_error) {
    console.error(`Error processing beatmap`, _error);
    const folderName = basename(metaList[0].beatmap_path);
    console.log(folderName)
    updateMeta2(metaList, UNKNOWN_ERROR);
  }
}

function logPath(beatmapPath: string, difficulty: string){
  // log the changes
  console.log("*************************");
  console.log("Handling:");
  const folderName = basename(beatmapPath);
  const command1 = `Z:\\Beatmap\\${folderName}`
  const command2 = `Remove-Item -Path "Z:\\Beatmap_Debug\\*" -Recurse -Force`
  const command3 = `robocopy "Z:\\Beatmap\\${folderName}" "Z:\\Beatmap2\\${folderName}" /E`;
  const command4 = `robocopy "Z:\\Beatmap\\${folderName}\\processed" "Z:\\Beatmap_Debug\\${folderName}\\processed" /E`;
  console.log(command1);
  console.log(difficulty)
  console.log(command2);
  console.log(command3); 
  console.log(command4);
}

const processDirectory = async (dir: { name: string, fullPath: string }) => {
  try {
    const infoPath = await getInfoPath(dir.fullPath);
    const info = bsmap.readInfoFileSync(`${dir.fullPath}/${infoPath}`);
    const difficultyTuples = getDifficultyTuples(info);
    for (const difficultyTuple of difficultyTuples) {
      updateMeta(infoPath, info, difficultyTuple, dir);
    }
  } catch (_error) {
    result[UNKNOWN_ERROR].push(dir.name);
    logPath(dir.fullPath, "");
    console.error("Error processing directory:", dir.name, _error);
  }
};

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
function isSupportedNote(time: number, tolerance = 0.099): boolean {
  /**
   * Checks if the given time is a multiple of 1/6 or 1/8 note within a small tolerance.
   *
   * @param time - The time in beats to check.
   * @param tolerance - Allowed floating-point precision error.
   * @returns True if the time is a multiple of 1/6 or 1/8, otherwise False.
   */
  let scaledTime = time * 8;
  const isOneEigth = Math.abs(Math.round(scaledTime) - scaledTime) < tolerance;
  scaledTime = time * 8;
  const isSixth = Math.abs(Math.round(scaledTime) - scaledTime) < tolerance;
  return isOneEigth || isSixth
}

function getComplexBeats(difficultyFile: IWrapBeatmap): number[] {
  const beats = difficultyFile.colorNotes.map(note => note.time);
  return beats.filter(time => !isSupportedNote(time));
}

async function copyDifficulty(filePath: string, targetPath: string) {
  const targetString: string[] = ["_version","version"];    
  try {
      console.log(`Copying ${filePath} to ${targetPath}`);
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
      console.log(`Copied ${filePath} to ${targetPath}`);
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
  console.log('create processed folder:', processedPath);
  //copy info.dat processed folder
  // await Deno.copyFile(`${meta.beatmap_path}/${meta.info_name}`, `${processedPath}/${meta.info_name}`);
  
}

function getOffset(noteList: number[], alignmentNote: number){
  const offsets = noteList.map(time => parseFloat((time % alignmentNote).toFixed(3)));
  const countMap = new Map<number, number>();
  for (const num of offsets) {
    countMap.set(num, (countMap.get(num) || 0) + 1);
  }
  // 找到出现最多的数将其作为offset
  let maxCountNum: number | null = null;
  let maxCount = 0;
  for (const [num, count] of countMap) {
    if (count > maxCount) {
      maxCount = count;
      maxCountNum = num;
    }
  }
  //sample
  //countMap sample: Map(2) {size: 2, 0.172 => 478, 0.047 => 88}
  //minCountNum: [0.047]
  //minCount: 88
  //maxCountNum: 0.172
  //maxCount: 478
  let positiveOffset = maxCountNum as number
  let negativeOffset = -(alignmentNote - (maxCountNum as number))
  if(maxCountNum === 0){
    positiveOffset = alignmentNote
    negativeOffset = -alignmentNote
  }
  //negative offset has higher priority
  return [positiveOffset, negativeOffset]
}
function handleOffset(difficultyFileList: IWrapBeatmap[], meta: any){
  const colorNoteList = difficultyFileList.map(difficultyFile => difficultyFile.colorNotes.map(note=>note.time))
  const noteList = [...new Set(colorNoteList.flat())];
  //origin complex beat
  const complexbeats = noteList.filter(time => !isSupportedNote(time));
  //bigger alignment note has high priority
  const alignmentNoteList = [0.125, 0.25, 0.5]
  let offsetList: number[] = []
  for (const alignmentNote of alignmentNoteList)
    offsetList = offsetList.concat(getOffset(noteList, alignmentNote))
  //complex beat without potential offset
  const noteListWithoutOffset = offsetList.map(offset=>noteList.map(time=>time-offset))
  const complexBeatsWithoutOffset = noteListWithoutOffset.map(noteList=>noteList.filter(time => !isSupportedNote(time)))
  // 找到出现次数最少的数作为offset
  let minCountNum;
  let minCount = Infinity;
  for (const [index, complexBeat] of complexBeatsWithoutOffset.entries()) {
    const count = complexBeat.length
    if (count<=minCount){
      minCount = count;
      minCountNum = offsetList[index]; // 重新赋值
    }
  }
  //show stats
  const potentialOffsetRatio = (complexbeats.length - minCount) / noteList.length
  console.log(`origin complex beat length: ${complexbeats.length}`)
  console.log(`potential offset: ${offsetList}`)
  console.log(`complex beat length after removing potential:`)
  console.log(complexBeatsWithoutOffset.map(noteList=>noteList.length))
  console.log(`min Count number: ${minCountNum} (count: ${minCount})`)
  console.log(`potentialOffsetRatio: ${potentialOffsetRatio}`)
  //有offset，其等于minCountNum
  let offset = 0
  if (minCount < complexbeats.length && potentialOffsetRatio >= offsetThreshold ){
    //set offset
    offset = (minCountNum as number);
    // remove offset from difficulty.colorNotes
    difficultyFileList.map(difficultyFile=>difficultyFile.colorNotes.map(note => {note.time = note.time - offset;}));
    offset = offset * 60 / meta.bpm * 1000;
    // update offset to info.dat
    // if want to update offset, edit it to +=offset
  //   const processedPath = `${meta.beatmap_path}/processed`;
  //   const info = bsmap.readInfoFileSync(`${processedPath}/${meta.info_name}`);
  //   info.difficulties.map(difficulty => {
  //     if (difficulty.characteristic === 'Standard') {
  //       difficulty.customData._editorOffset = offset;
  //       difficulty.customData._editorOldOffset = offset;
  //     }
  //   });
  //   bsmap.writeInfoFile(info, 2, {
  //     directory: processedPath,
  //     filename: meta.info_name
  //   });
  }
  return offset;
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
      if (note.time > lastNote.time && (note.time - lastNote.time < slideNote) && (lastNote.direction == note.direction || note.direction == 8)){
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
  console.log(`Note with slide (length:,${noteWithSlide.length}):`)
  // console.log(noteWithSlide)
  console.log()
}

function updateMeta(infoPath: string, info: IWrapInfo, difficultyTuple: [string, string, number, number, number], dir: { name: string, fullPath: string }) {
  output_meta.push({
    id: `${dir.name}_${difficultyTuple[1]}`,
    beatmap_path: dir.fullPath,
    info_name: infoPath,
    song_name: info.audio.filename,
    bpm: info.audio.bpm,
    beatmap_name: difficultyTuple[0],
    difficulty: difficultyTuple[1],
    njs: difficultyTuple[2],
    njsoffset: difficultyTuple[3],
    editor_offset: difficultyTuple[4],
    audio_offset: info.audio.audioOffset,
    status: RAW_DATA,
  });
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
const [directory, manifest_directory, pipeline, target] = Deno.args;

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
const slideNote = 0.125
const offsetThreshold = 0.1
const processedThreshold = 10

const difficulty_cache: [string, string][] = []

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
else {
  const input_meta: JSON[] = await loadJsonl(manifest_directory)
  const input_meta_group: Record<string, any[]> = {};
  for (const meta of input_meta) {
    const currentPath: string = (meta as any).beatmap_path;
    (input_meta_group[currentPath] ||= []).push(meta);
  }
  if (pipeline === "copy_beatmap"){
    // copy beatmap files to processed folder, and save all beatmap changes there
    await Promise.all(Object.entries(input_meta_group).map(([key, value])=>copyBeatmap(value)));
    const limit = pLimit(1);
    await Promise.all(input_meta.map((meta: any)=> limit(() => copyDifficulty(`${meta.beatmap_path}/${meta.beatmap_name}`, `${meta.beatmap_path}/processed/${meta.beatmap_name}`))));
    output_meta.push(...input_meta)
  }
  else if (pipeline === "process_beatmap"){
    // await Promise.all(Object.entries(input_meta_group).map(([key, value])=>processBeatmap(value)));
    const input_meta_list = Object.values(input_meta_group);
    const batchSize = 1000;
    for (let i = 0; i < input_meta_list.length; i += batchSize) {
      const sub_list = input_meta_list.slice(i, i + batchSize);
      await Promise.all(sub_list.map(element => processBeatmap(element)));
      for (const [difficultyPath, jsonData] of difficulty_cache)
        await Deno.writeTextFile(difficultyPath, jsonData);
      difficulty_cache.length = 0;

    }
    
    
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
}



const end = performance.now();
console.log(`代码执行时间: ${((end - start)/1000).toFixed(2)} 秒`);

// Save metadata to file
const fileContent = output_meta.map((obj) => JSON.stringify(obj)).join("\n");
await Deno.writeTextFile(manifest_directory, fileContent);


