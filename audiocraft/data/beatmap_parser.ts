//deno run --allow-net --allow-read --allow-write beatmap_parser.ts 
//deno run --allow-net --allow-read --allow-write beatmap_parser.ts &

import { Application, Router, Request, Response, Context } from "https://deno.land/x/oak/mod.ts";
import * as bsmap from '../../../BeatSaber-JSMap/src/mod.ts';
import { join, basename, dirname } from "https://deno.land/std@0.201.0/path/mod.ts";
import { exists, copy } from "https://deno.land/std/fs/mod.ts";






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

// 拷贝目录
async function copyDirectory(source: string, destination: string) {
  await copy(source, destination, { overwrite: true });
}

// 生成新目录名
function generateNewDirectoryName(directory: string): string {
  const parentDir = dirname(directory);
  const baseName = basename(directory);
  return join(parentDir, `${baseName}_copy`);
}

export const read = async (
  { request, response }: { request: Request; response: Response },
) => {
  const queryParams = request.url.searchParams;
  const directory = queryParams.get("directory");  // Fetch the 'name' parameter from the URL
  const write_parse_switch: boolean = queryParams.get("write_parse_switch")==="True";
  const complex_beat_number = parseFloat(queryParams.get("complex_beat_number")||"0.125");
  if (!directory) {
    response.status = 400;
    response.body = "Directory path are required.";
    return;
  }
  // 自动生成新目录名
  // const newDirectory = generateNewDirectoryName(directory);
  // await copyDirectory(directory, newDirectory);  // 拷贝到新目录
  const directories = await getDirectoriesWithPaths(directory||'');
  
  const output_meta = [];
  let load: number = 0;
  const fail_dir_error: string[] = [];
  const fail_audio_offset: string[] = [];
  const fail_bpm_events: string[] = [];
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
      const difficultyTuples: [string, string,number,number][] = info.difficulties
        .filter(difficulty => difficulty.characteristic === 'Standard')  // 过滤characteristic为Standard的
        .map(difficulty => [difficulty.filename, difficulty.difficulty,difficulty.njs,difficulty.njsOffset]);  // 提取filename,difficulty等组成元组

      const songPath = dir.fullPath+"/"+info.audio.filename;
      
      for (const difficultyTuple of difficultyTuples){
        const difficultyFile = bsmap.readDifficultyFileSync(difficultyTuple[0]);
        // filter map with bpmEvents
        if (difficultyFile.bpmEvents.length != 0){
          //例外：如果只有一个bpm event且其time=0并bpm和本身bpm一样
          if (!(difficultyFile.bpmEvents.length == 1 && difficultyFile.bpmEvents[0].time ==0 && difficultyFile.bpmEvents[0].bpm == info.audio.bpm)){
            fail_bpm_events.push(dir.name+'_'+difficultyTuple[1])
            continue;
          }
        }
        // filter 找出需要16分音符或者更小单位的时间
        const beat: number[] = difficultyFile.colorNotes.map(colorNote => colorNote.time);
        const complexBeats = beat.filter(time => time % complex_beat_number !== 0);
        if (complexBeats.length != 0){
          fail_complex_beats.push(dir.name+'_'+difficultyTuple[1]);
          continue;
        }
        const jsonData = JSON.stringify(difficultyFile);
        const difficultyPath = dir.fullPath+"/"+difficultyTuple[1]+".json"
        if (write_parse_switch){
          await Deno.writeTextFile(difficultyPath, jsonData);
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
        output_meta.push({id: id, beatmap_info_path: join(dir.fullPath, info_path), song_path: songPath, beatmap_file_path : difficultyPath, difficulty : difficultyTuple[1], bpm : info.audio.bpm, njs : difficultyTuple[2], njsoffset : difficultyTuple[3]})
        load++;
      }
      
    } catch (error) {
      console.error("捕获到的错误:", error);
      fail_dir_error.push(dir.name);
    }
  }

  // summary
  response.status = 200;
  response.body = { 
    dir_num : directories.length,
    total_num : load+fail_dir_error.length+fail_audio_offset.length+fail_bpm_events.length+fail_complex_beats.length,
    load : load,
    fail_dir_error_number: fail_dir_error.length,
    fail_audio_offset_number: fail_audio_offset.length,
    fail_bpm_events_number: fail_bpm_events.length,
    fail_complex_beats_number: fail_complex_beats.length,
    fail_dir_error: fail_dir_error,
    fail_audio_offset: fail_audio_offset,
    fail_bpm_events: fail_bpm_events,
    fail_complex_beats: fail_complex_beats,
    output_meta: output_meta // 直接将 output_meta 数据添加到响应体中
   };
}
export const generate_difficulty = async (
  { request, response }: { request: Request; response: Response },
) => {
  const queryParams = request.url.searchParams;
  const filePath = queryParams.get("beatmap_file_path")||'';
  const difficulty = queryParams.get("difficulty")|| '';
  const save_directory = queryParams.get("save_directory")|| '';
  const difficulty_version = parseInt(queryParams.get("difficulty_version")||'3');
  const beatmapInfoPath = queryParams.get("beatmap_info_path")||'';
  const info_version = parseInt(queryParams.get("info_version")||'2');
  const beatmapName = queryParams.get("beatmap_name")||'';
  // bsmap.globals.directory = save_directory|| '';
  let parsedDifficulty, difficultyFile
  try {
    difficultyFile = await Deno.readTextFile(filePath);
  } catch (error) {    
    response.status = 400;
    response.body = { error: "JSON not found" };
    return;
  }
  try {
    parsedDifficulty = JSON.parse(difficultyFile); // Parse the string as JSON
  } catch (error) {    
    response.status = 400;
    response.body = { error: "Invalid JSON format" };
    return;
  }
  const difficultyData = new bsmap.Beatmap
  parsedDifficulty.difficulty.colorNotes.forEach((colorNote: any) => {
      const colorNotes = bsmap.ColorNote.create(colorNote);
      difficultyData.colorNotes.push(...colorNotes);
    });
  await bsmap.writeDifficultyFile(difficultyData, difficulty_version, {
    directory: save_directory,
    filename: difficulty+"Standard.dat"
  });

  //move info
  bsmap.globals.directory = dirname(beatmapInfoPath)
  const info = bsmap.readInfoFileSync(basename(beatmapInfoPath)); 
  // writeInfoFileSync(info)
  info.song.title = beatmapName
  info.audio.filename = "song.ogg"
  info.difficulties = info.difficulties.filter(difficulties => difficulties.characteristic === 'Standard' && difficulties.difficulty === difficulty)
  info.difficulties[0].filename = difficulty+"Standard.dat"

  await bsmap.writeInfoFile(info, info_version, {
    directory: save_directory,
    filename: 'Info.dat'
  })

  response.status = 200;
  response.body = {
    success: true
  };
};

const router = new Router();
router
.get("/read", read)
.get("/generate_difficulty", generate_difficulty)
  



const app = new Application();
app.use(router.routes());
app.use(router.allowedMethods());

await app.listen({ port: 8000 });