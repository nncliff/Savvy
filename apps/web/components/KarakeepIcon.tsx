import KarakeepFull from "@/public/icons/savvy-full.svg";

export default function KarakeepLogo({ height }: { height: number }) {
  return (
    <span className="flex items-center">
      <KarakeepFull height={height} className={`fill-foreground`} />
    </span>
  );
}
