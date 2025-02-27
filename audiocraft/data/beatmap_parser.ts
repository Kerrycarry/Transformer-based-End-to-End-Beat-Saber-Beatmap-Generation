//deno run --allow-net --allow-read --allow-write beatmap_parser.ts 
//deno run --allow-net --allow-read --allow-write beatmap_parser.ts &

import { Application, Router, Request, Response } from "https://deno.land/x/oak/mod.ts";
import * as bsmap from '../../../BeatSaber-JSMap/src/mod.ts';
import { basename, dirname } from "https://deno.land/std@0.201.0/path/mod.ts";

export const generate_difficulty = async (
  { request, response }: { request: Request; response: Response },
) => {
  const queryParams = request.url.searchParams;
  const filePath = queryParams.get("processed_beatmap_json")||'';
  const difficulty = queryParams.get("difficulty")|| '';
  const save_directory = queryParams.get("save_directory")|| '';
  const difficulty_version = parseInt(queryParams.get("difficulty_version")||'3');
  const beatmapInfoPath = queryParams.get("info_name")||'';
  const info_version = parseInt(queryParams.get("info_version")||'2');
  const beatmapName = queryParams.get("beatmap_name")||'';
  const write_info_switch: boolean = queryParams.get("write_info_switch")==="True";
  // bsmap.globals.directory = save_directory|| '';
  let parsedDifficulty, difficultyFile
  // generate difficulty dat file
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
  // generate info dat file
  if (write_info_switch){
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
  }

  response.status = 200;
  response.body = {
    success: true
  };
};

const router = new Router();
router
.get("/generate_difficulty", generate_difficulty)

const app = new Application();
app.use(router.routes());
app.use(router.allowedMethods());

await app.listen({ port: 8000 });