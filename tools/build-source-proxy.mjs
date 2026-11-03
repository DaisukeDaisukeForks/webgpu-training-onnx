// Bootstrap a new Site from the source ZIP before acquiring any teacher payload.
import fs from 'node:fs';
import path from 'node:path';
const root=path.dirname(path.dirname(new URL(import.meta.url).pathname)),dist=path.join(root,'dist');
fs.rmSync(dist,{recursive:true,force:true});fs.mkdirSync(path.join(dist,'client'),{recursive:true});fs.mkdirSync(path.join(dist,'server'),{recursive:true});fs.mkdirSync(path.join(dist,'.openai'),{recursive:true});
fs.copyFileSync(path.join(root,'.openai/hosting.json'),path.join(dist,'.openai/hosting.json'));
const assets=JSON.parse(fs.readFileSync(path.join(root,'config/source-assets.json'),'utf8'));
fs.writeFileSync(path.join(dist,'server/index.js'),fs.readFileSync(path.join(root,'worker/index.mjs'),'utf8')+`\nexport default handler(new Set(),${JSON.stringify(assets)});\n`);
fs.writeFileSync(path.join(dist,'client/index.html'),'<!doctype html><html lang="ja"><meta charset="utf-8"><title>教師データの準備</title><p>教師データのキャッシュ経路を準備しました。取得後、学習アプリをビルド・公開してください。</p></html>');
console.log('Small Sites source proxy built; no teacher files, images or R2 teacher dependency.');
