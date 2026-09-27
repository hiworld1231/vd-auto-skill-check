const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');

function bench(){
  const html=fs.readFileSync(__dirname+'/skillcheck.html','utf8');
  const script=html.match(/<script>([\s\S]*?)<\/script>/)[1];
  let now=0,frame,timer,keydown;
  const nodes=new Map();
  const document={getElementById(id){
    if(!nodes.has(id)) nodes.set(id,{style:{},textContent:'',setAttribute(){}});
    return nodes.get(id);
  }};
  const window={};
  vm.runInNewContext(script,{document,window,performance:{now:()=>now},
    addEventListener:(name,cb)=>{if(name==='keydown')keydown=cb},
    requestAnimationFrame:cb=>{frame=cb},
    setTimeout:(cb,delay)=>{timer={cb,at:now+delay};return timer},
    clearTimeout:id=>{if(timer===id)timer=null}});
  return {
    state:()=>window.VDBenchState(),
    advance(ms){now+=ms;if(timer&&timer.at<=now){const cb=timer.cb;timer=null;cb()}if(frame)frame(now)},
    key(code){keydown({code,repeat:false,preventDefault(){}})},
    result:()=>document.getElementById('result').textContent,
  };
}

test('bench starts ordinary checks and Frenzy keeps the rotating phase',()=>{
  const b=bench();
  b.advance(600);
  assert.equal(b.state().count,1);
  const before=b.state().phase;
  b.key('KeyF');
  assert.equal(b.state().count,2);
  assert.equal(b.state().frenzy,true);
  assert.equal(b.state().phase,before);
  b.advance(150);
  assert.ok(b.state().phase>before);
});

test('a miss stops Frenzy and a new ordinary check can start',()=>{
  const b=bench();
  b.key('KeyF');
  assert.equal(b.state().count,1);
  b.key('Space');
  assert.match(b.result(),/MISS/);
  assert.equal(b.state().frenzy,false);
  b.advance(900);
  assert.equal(b.state().count,2);
});
