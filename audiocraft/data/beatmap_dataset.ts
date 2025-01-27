//deno run --allow-net --allow-read --allow-write beatmap_parser.ts 
//deno run --allow-net --allow-read --allow-write beatmap_parser.ts &

import * as bsmap from '../../../BeatSaber-JSMap/src/mod.ts';
import { join, basename, dirname } from "https://deno.land/std@0.201.0/path/mod.ts";
import { exists, copy } from "https://deno.land/std/fs/mod.ts";

const pathCount: Record<string, number> = {};
let complexCount : number = 0

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
  if (numStr.includes('.')) {
    return numStr.split('.')[1].length;
  }
  return 0; // 没有小数点，返回0
}

async function getDirectoriesWithPaths(dirPath: string) {
  const entries = [];
  // 遍历目录中的文件和文件夹
  for await (const entry of Deno.readDir(dirPath)) {
    if (entry.isDirectory) {
      // 获取文件夹的完整路径
      const fullPath = join(dirPath, entry.name);
      entries.push({ name: entry.name, fullPath });
    }
  }
  return entries;
}

const directory = Deno.args[0];
const manifest_directory = Deno.args[1];
const pipeline = Deno.args[2];
const complex_beat_number = Number(Deno.args[3]);

const directories = await getDirectoriesWithPaths(directory||'');
const new_entries = [];

const output_meta = [];
let load: number = 0;
const fail_dir_error: string[] = [];
const fail_audio_offset: string[] = [];
const fail_editor_offset: string[] = [];
const fail_bpm_events: string[] = [];
const fail_no_offset: string[] = [];
const fail_floating_error: string[] = [];
const fail_complex_beats: string[] = [];
for (const dir of directories){
  bsmap.globals.directory = dir.fullPath || '';
  try {
    let info_path = "Info.dat"
    const file1 = `${dir.fullPath}/info.dat`;
    if (await exists(file1)) {
      info_path = "info.dat"
    }
    const info = bsmap.readInfoFileSync(info_path); 
    // filter map with audioOffset
    if (info.audio.audioOffset != 0){
      fail_audio_offset.push(dir.name);
      continue; 
    }
    //parse 不同的 diff 为json并且保存
    const difficultyTuples: [string, string,number,number, number][] = info.difficulties
      .filter(difficulty => difficulty.characteristic === 'Standard')  // 过滤characteristic为Standard的
      .map(difficulty => [difficulty.filename, difficulty.difficulty,difficulty.njs,difficulty.njsOffset, difficulty.customData._editorOffset]);  // 提取filename,difficulty等组成元组
    
    for (const difficultyTuple of difficultyTuples){
      const difficultyFile = bsmap.readDifficultyFileSync(difficultyTuple[0]);
      // filter map with bpmEvents
      const res = hasBpmFieldWithNonEmptyListRecursive(difficultyFile)
      if (res){
          fail_bpm_events.push(dir.name+'_'+difficultyTuple[1])
          continue;
      }
      // if (difficultyFile.bpmEvents.length != 0){
      //   //例外：如果只有一个bpm event且其time=0并bpm和本身bpm一样
      //   if (!(difficultyFile.bpmEvents.length == 1 && difficultyFile.bpmEvents[0].time ==0 && difficultyFile.bpmEvents[0].bpm == info.audio.bpm)){
      //     fail_bpm_events.push(dir.name+'_'+difficultyTuple[1])
      //     continue;
      //   }
      // }
      // if (difficultyFile.difficulty?.customData?._BPMChanges && difficultyFile.difficulty.customData._BPMChanges.length > 0){
      //     fail_bpm_events.push(dir.name+'_'+difficultyTuple[1])
      //     continue;
      //   }
      if (difficultyTuple[4] != 0){
          fail_editor_offset.push(dir.name+'_'+difficultyTuple[1])
          continue;
        }

      // filter 找出需要16分音符或者更小单位的时间
      const beat: number[] = difficultyFile.colorNotes.map(colorNote => colorNote.time);
      const complexBeats = beat.filter(time => time % complex_beat_number !== 0);
      if (complexBeats.length != 0){
        const timeDeciamlPlaces: number[] = complexBeats.map(time => getDecimalPlaces(time));
        if (timeDeciamlPlaces.some(num => num > 6)){
          fail_floating_error.push(dir.name+'_'+difficultyTuple[1]);
          continue;
        }
        const offsets: number[] = complexBeats.map(time => time % complex_beat_number);
        if(offsets.every((val, _, arr) => val === arr[0])){
          fail_no_offset.push(dir.name+'_'+difficultyTuple[1]);
          continue;
        }
        if (complexBeats.length/difficultyFile.colorNotes.length > 0.2){
          console.log("found > 0.2")
          console.log(dir.name+'_'+difficultyTuple[1])
          console.log("*******************",complexBeats.length, difficultyFile.colorNotes.length, complexBeats.length/difficultyFile.colorNotes.length)
          console.log(difficultyFile.colorNotes.map(note => note.time))
          complexCount = complexCount + 1
        }
        console.log("*******************",complexBeats)
        
        new_entries.push({name: dir.name, fullPath: dir.fullPath})
        fail_complex_beats.push(dir.name+'_'+difficultyTuple[1]);
        continue;
      }
      //更新meta, egg, diff json位置，bpm, njs, njsoffset
      const regex = /^[a-zA-Z0-9]+/;
      const match = dir.name.match(regex);
      let id
      if (match) {
        id = match[0]
      }
      else{
        id = dir.name
      }
      output_meta.push({
        id: id + "_" + difficultyTuple[1],
        beatmap_path: dir.fullPath,
        info_name: info_path,
        song_name: info.audio.filename,
        beatmap_file_name: difficultyTuple[1]+".json",
        difficulty: difficultyTuple[1],
        bpm: info.audio.bpm,
        njs: difficultyTuple[2],
        njsoffset: difficultyTuple[3],
        note_num: {
          colorNotes: difficultyFile.colorNotes.length,
          bombNotes: difficultyFile.bombNotes.length,
          obstacles: difficultyFile.obstacles.length,
          arcs: difficultyFile.arcs.length,
          chains: difficultyFile.chains.length,
        },
      });
      load++;
    }
    
  } catch (error) {
    console.error("捕获到的错误:", error);
    fail_dir_error.push(dir.name);
  }
}

console.log("*****************************result:")
console.log("Path count:", pathCount);
console.log("Nontcomplex count:", complexCount);


// Convert each JSON object to a string and write it to the file
const fileContent = output_meta.map((obj) => JSON.stringify(obj)).join("\n");
await Deno.writeTextFile(manifest_directory, fileContent);

// summary
const res = { 
  dir_num : directories.length,
  total_num : load+fail_dir_error.length+fail_audio_offset.length+fail_bpm_events.length+fail_complex_beats.length,
  load : load,
  fail_dir_error_number: fail_dir_error.length,
  fail_audio_offset_number: fail_audio_offset.length,
  fail_editor_offset_number : fail_editor_offset.length,
  fail_bpm_events_number: fail_bpm_events.length,
  fail_no_offset_number: fail_no_offset.length,
  fail_floating_error_number: fail_floating_error.length,
  fail_complex_beats_number: fail_complex_beats.length,
  fail_dir_error: fail_dir_error,
  fail_audio_offset: fail_audio_offset,
  fail_editor_offset : fail_editor_offset,
  fail_bpm_events: fail_bpm_events,
  fail_complex_beats: fail_complex_beats,
  };
console.log(res);
